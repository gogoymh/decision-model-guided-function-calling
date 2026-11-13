package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static io.github.gogoymh.dmcall.Support.Mock;
import static io.github.gogoymh.dmcall.Support.SECRET;
import static io.github.gogoymh.dmcall.Support.answer;
import static io.github.gogoymh.dmcall.Support.exampleMessages;
import static io.github.gogoymh.dmcall.Support.exampleTools;
import static io.github.gogoymh.dmcall.Support.seoul;
import static io.github.gogoymh.dmcall.Support.weather;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Jev and the PPLX Decider: the wire format, verdicts, keys, thresholds and failures (test_decision_models.py). */
class DecisionModelTest {
    private static final String REVISION = "14c9d90137962d300f3de95633f7b6535e80c5ce";

    @BeforeEach
    void isolate() {
        Support.isolate();
    }

    private static Jev jev(Mock mock) {
        return Jev.builder().apiKey("jev-key").transport(mock).sleeper(Support.NO_SLEEP).build();
    }

    private static ObjectNode provenance() {
        ObjectNode body = Json.object();
        body.put("model_id", "perplexity-ai/pplx-decider-v1.1-27b");
        body.put("revision", REVISION);
        body.put("native_attention", "noncausal_full_attention");
        body.put("max_input_tokens", 8192);
        body.put("temperature", 1.0087417621345625);
        return body;
    }

    @Test
    void wireFormatSendsOnlyObservableInputs() {
        Mock mock = Mock.serving(answer(0.9, 0.8));
        ArrayNode tools = Json.array();
        ObjectNode tool = weather();
        tool.putObject("metadata").put("reference", "get_weather(city='Seoul')");
        tools.add(tool);
        ArrayNode messages = exampleMessages();
        messages.addObject().put("role", "user").put("content", "hi").put("ground_truth", "nope");
        Judgment record = jev(mock).judge(messages, tools, seoul());
        JsonNode body = mock.body(0);
        assertEquals("https://api.typesafe.ai/v1/systemone", mock.requests.get(0).url());
        assertEquals("jev-1.13.0", body.get("model").textValue());
        assertEquals(Json.parse("{\"role\": \"user\", \"content\": \"hi\"}"),
                body.get("state").get("messages").get(2));
        ArrayNode expectedTools = Json.array();
        expectedTools.add(weather().get("function"));
        assertEquals(expectedTools, body.get("state").get("tools"));
        assertEquals(Calls.toJson(seoul()), body.get("state").get("proposed_calls"));
        ObjectNode questions = Json.object();
        questions.set("function_correct", Prompts.functionQuestion());
        questions.set("arguments_correct", Prompts.argumentQuestion());
        assertEquals(questions, body.get("questions"));
        assertEquals(Boolean.TRUE, record.accepted());
        assertEquals(0.9, record.functionProbability());
        assertEquals(0.8, record.argumentsProbability());
        assertTrue(record.feedback().get("failed_checks").isEmpty());
        assertEquals(1, record.apiCalls());
    }

    @Test
    void noCallAsksOnlyTheFunctionQuestion() {
        Mock mock = Mock.serving(answer(0.97, null));
        Judgment record = jev(mock).judge(exampleMessages(), exampleTools(), List.of());
        List<String> asked = new ArrayList<>();
        mock.body(0).get("questions").fieldNames().forEachRemaining(asked::add);
        assertEquals(List.of("function_correct"), asked);
        assertEquals(Boolean.TRUE, record.accepted());
        assertTrue(record.argumentsSkipped());
        assertNull(record.argumentsProbability());
        assertTrue(record.feedback().get("checks").get("arguments").get("pass").isNull());
    }

    @Test
    void bothQuestionsMustPassAtTheThreshold() {
        Object[][] cases = {{0.4, 0.9, List.of("function_selection")}, {0.9, 0.49, List.of("arguments")},
                {0.5, 0.5, List.of()}};
        for (Object[] c : cases) {
            Judgment record = jev(Mock.serving(answer((Double) c[0], (Double) c[1]))).judge(exampleMessages(),
                    exampleTools(), seoul());
            assertEquals(Json.node(c[2]), record.feedback().get("failed_checks"));
            assertEquals(((List<?>) c[2]).isEmpty(), record.accepted());
        }
    }

    @Test
    void schemaFailureIsJudgedButCannotBeOverruled() {
        Mock mock = Mock.serving(answer(0.99, 0.99));
        Judgment record = jev(mock).judge(exampleMessages(), exampleTools(),
                List.of(Call.of("get_weather", Map.of())));
        assertEquals(1, mock.requests.size());
        assertEquals(0.99, record.argumentsProbability());  // a readable candidate is still judged
        assertEquals(Boolean.FALSE, record.argumentsPass());
        assertEquals(Boolean.FALSE, record.accepted());
        assertFalse(record.schemaErrors().isEmpty());
    }

    @Test
    void unreadableCandidateIsNotSent() {
        for (List<Call> calls : java.util.Arrays.asList(null, List.of(new Call("get_weather", null)))) {
            Mock mock = new Mock(request -> {
                throw new AssertionError("An unreadable candidate must not reach the decision model");
            });
            Judgment record = jev(mock).judge(exampleMessages(), exampleTools(), calls);
            assertEquals(0, record.apiCalls());
            assertNull(record.functionProbability());
            assertEquals(Boolean.FALSE, record.accepted());
        }
    }

    @Test
    void thresholdsKeysAndModel() {
        IllegalArgumentException missing = assertThrows(IllegalArgumentException.class, () -> new Jev());
        assertTrue(missing.getMessage().contains("TYPESAFE_API_KEY"));
        Support.environment(Map.of("TYPESAFE_API_KEY", SECRET, "TYPESAFE_MODEL", "jev-2"));
        Mock mock = Mock.serving(answer(0.75, 0.85));
        Jev model = Jev.builder().threshold(0.7).argumentsThreshold(0.9).transport(mock).build();
        Judgment record = model.judge(exampleMessages(), exampleTools(), seoul());
        assertEquals("Bearer " + SECRET, mock.requests.get(0).header("authorization"));
        assertFalse(model.toString().contains(SECRET));
        assertFalse(mock.requests.get(0).toString().contains(SECRET));
        assertEquals("jev-2", mock.body(0).get("model").textValue());
        assertEquals(Boolean.TRUE, record.functionPass());
        assertEquals(Boolean.FALSE, record.argumentsPass());
        for (double bad : new double[]{1.5, -0.1, Double.NaN}) {
            IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                    () -> Jev.builder().apiKey("k").threshold(bad).build());
            assertTrue(error.getMessage().contains("Thresholds"));
        }
    }

    @Test
    void retriesAreBoundedAndErrorsSanitized() {
        Mock retried = Mock.replies(HttpTransport.Response.of(429, null, Map.of("Retry-After", "0")),
                HttpTransport.Response.json(200, answer(0.95, 0.95)));
        assertEquals(2, jev(retried).judge(exampleMessages(), exampleTools(), seoul()).apiCalls());
        Mock failing = Mock.always(HttpTransport.Response.of(401, SECRET, Map.of("x-request-id", "req-7")));
        DecisionModelException caught = assertThrows(DecisionModelException.class,
                () -> jev(failing).judge(exampleMessages(), exampleTools(), seoul()));
        assertEquals(1, failing.requests.size());
        assertEquals(401, caught.statusCode());
        assertEquals("req-7", caught.requestId());
        assertFalse(caught.getMessage().contains(SECRET));
        Mock overloaded = Mock.always(HttpTransport.Response.of(503, null, null));
        caught = assertThrows(DecisionModelException.class, () -> Jev.builder().apiKey("jev-key").maxRetries(2)
                .transport(overloaded).sleeper(Support.NO_SLEEP).build()
                .judge(exampleMessages(), exampleTools(), seoul()));
        assertEquals(3, caught.attempts());
        Mock down = new Mock(request -> null);
        caught = assertThrows(DecisionModelException.class, () -> jev(down).judge(exampleMessages(), exampleTools(),
                seoul()));
        assertEquals("connection_failed", caught.code());
        assertEquals(4, down.requests.size());
    }

    @Test
    void retryAfterDelaysAreBounded() {
        List<Double> waits = new ArrayList<>();
        Mock mock = Mock.replies(HttpTransport.Response.of(429, null, Map.of("Retry-After", "120")),
                HttpTransport.Response.of(503, null, Map.of("Retry-After", "soon")),
                HttpTransport.Response.json(200, answer(0.95, 0.95)));
        Jev.builder().apiKey("k").transport(mock).sleeper(waits::add).build()
                .judge(exampleMessages(), exampleTools(), seoul());
        assertEquals(List.of(30.0, 2.0), waits);
    }

    @Test
    void invalidResponsesAreErrorsNotRejections() {
        ObjectNode noUsage = answer(0.95, 0.95);
        noUsage.putNull("usage");
        ObjectNode noModel = answer(0.95, 0.95);
        noModel.put("model", "");
        JsonNode[] bodies = {Json.parse("{\"answers\": {}}"), answer(1.5, 0.95), noUsage, noModel,
                Json.parse("[\"not\", \"an\", \"object\"]")};
        for (JsonNode body : bodies) {
            DecisionModelException error = assertThrows(DecisionModelException.class,
                    () -> jev(Mock.always(HttpTransport.Response.json(200, body))).judge(exampleMessages(),
                            exampleTools(), seoul()), body.toString());
            assertEquals("response_invalid", error.code());
        }
    }

    // PPLX Decider ----------------------------------------------------------------------
    private static PPLXDecider pplx(Mock mock, boolean checkProvenance) {
        return PPLXDecider.builder("http://127.0.0.1:8000").transport(mock).checkProvenance(checkProvenance).build();
    }

    @Test
    void pplxUsesTheSameQuestionsAndReportsTheCheckpoint() {
        Mock mock = Mock.serving(answer(0.9, 0.2, "autojev-qwen3.8-27b"));
        Judgment record = pplx(mock, false).judge(exampleMessages(), exampleTools(), seoul());
        JsonNode body = mock.body(0);
        assertEquals("http://127.0.0.1:8000/v1/systemone", mock.requests.get(0).url());
        assertEquals("jev-latest", body.get("model").textValue());
        assertEquals(Prompts.functionQuestion(), body.get("questions").get("function_correct"));
        assertEquals("perplexity-ai/pplx-decider-v1.1-27b", record.model());
        assertEquals(Boolean.FALSE, record.accepted());
        assertEquals("perplexity_decider_noul_and_deterministic_schema_validation",
                record.feedback().get("source").textValue());
        assertNull(mock.requests.get(0).header("authorization"));
    }

    @Test
    void pplxMissingUsageStaysUnknown() {
        ObjectNode body = Json.object();
        body.put("model", "autojev-qwen3.8-27b");
        body.set("answers", answer(0.95, 0.95).get("answers"));
        Judgment record = pplx(Mock.serving(body), false).judge(exampleMessages(), exampleTools(), seoul());
        Map<String, Long> unknown = new java.util.LinkedHashMap<>();
        unknown.put("input_tokens", null);
        unknown.put("output_tokens", null);
        assertEquals(unknown, record.usage());
    }

    @Test
    void pplxRejectsAnotherModel() {
        ObjectNode revised = answer(0.95, 0.95, "perplexity-ai/pplx-decider-v1.1-27b");
        revised.put("revision", "0".repeat(40));
        for (JsonNode body : new JsonNode[]{answer(0.95, 0.95, "some-other-model"), revised}) {
            DecisionModelException error = assertThrows(DecisionModelException.class,
                    () -> pplx(Mock.always(HttpTransport.Response.json(200, body)), false)
                            .judge(exampleMessages(), exampleTools(), seoul()));
            assertEquals("model_mismatch", error.code());
        }
    }

    @Test
    void pplxProvenanceCheckIsOptIn() {
        java.util.function.Function<HttpTransport.Request, HttpTransport.Response> handler = request ->
                request.url().endsWith("/provenance") ? HttpTransport.Response.json(200, provenance())
                        : HttpTransport.Response.json(200, answer(0.95, 0.95, "autojev-qwen3.8-27b"));
        Mock checked = new Mock(handler);
        PPLXDecider decider = pplx(checked, true);
        decider.judge(exampleMessages(), exampleTools(), seoul());
        decider.judge(exampleMessages(), exampleTools(), seoul());
        assertEquals(List.of("/provenance", "/v1/systemone", "/v1/systemone"), paths(checked));
        assertEquals("GET", checked.requests.get(0).method());
        Mock unchecked = new Mock(handler);
        pplx(unchecked, false).judge(exampleMessages(), exampleTools(), seoul());
        assertEquals(List.of("/v1/systemone"), paths(unchecked));
        ObjectNode main = provenance();
        main.put("revision", "main");
        DecisionModelException error = assertThrows(DecisionModelException.class,
                () -> pplx(Mock.always(HttpTransport.Response.json(200, main)), true)
                        .judge(exampleMessages(), exampleTools(), seoul()));
        assertEquals("provenance_mismatch", error.code());
    }

    private static List<String> paths(Mock mock) {
        List<String> paths = new ArrayList<>();
        mock.requests.forEach(request -> paths.add(request.url().substring("http://127.0.0.1:8000".length())));
        return paths;
    }

    @Test
    void pplxContextLimitIsAnError() {
        Mock over = Mock.always(HttpTransport.Response.json(422,
                Json.parse("{\"detail\": \"Prepared input exceeds the 8192-token limit.\"}")));
        DecisionModelException error = assertThrows(DecisionModelException.class,
                () -> pplx(over, false).judge(exampleMessages(), exampleTools(), seoul()));
        assertEquals("context_limit_exceeded", error.code());
        assertEquals(422, error.statusCode());
    }

    @ParameterizedTest
    @ValueSource(strings = {"http://10.0.0.4:8000", "https://user:pw@example.com", "http://localhost:8000/v1",
            "ftp://localhost"})
    void pplxRejectsUnsafeUrls(String url) {
        assertThrows(IllegalArgumentException.class, () -> new PPLXDecider(url));
    }

    @Test
    void pplxAcceptsLocalAndHttpsOrigins() {
        assertEquals("https://decider.example.com", new PPLXDecider("https://decider.example.com").baseUrl());
        assertEquals("http://localhost:8000", new PPLXDecider("http://localhost:8000/").baseUrl());
        assertEquals("http://[::1]:8000", new PPLXDecider("http://[::1]:8000").baseUrl());
    }

    @Test
    void genericSystemOneServer() {
        Mock mock = Mock.serving(answer(0.9, 0.9, "my-decider"));
        SystemOneDecisionModel model = SystemOneDecisionModel.builder("https://decider.internal/", "my-decider")
                .apiKey("k").transport(mock).build();
        Judgment record = model.judge(exampleMessages(), exampleTools(), seoul());
        assertEquals("https://decider.internal/v1/systemone", mock.requests.get(0).url());
        assertEquals("decision_model_noul_and_deterministic_schema_validation",
                record.feedback().get("source").textValue());
        assertThrows(IllegalArgumentException.class,
                () -> SystemOneDecisionModel.builder("https://x.example.com?q=1", "m").build());
    }

    @Test
    void askCanBeOverriddenForAnotherProtocol() {
        SystemOneDecisionModel custom = new SystemOneDecisionModel("https://unused.example.com", "local", null,
                SystemOneDecisionModel.builder("https://unused.example.com", "local").transport(Mock.always(null)),
                60.0, 0, true) {
            @Override
            public Answer ask(ObjectNode state, ObjectNode questions) {
                Map<String, Double> probabilities = new java.util.LinkedHashMap<>();
                questions.fieldNames().forEachRemaining(key -> probabilities.put(key, 0.8));
                return new Answer(probabilities, "local", Map.of("input_tokens", 1L, "output_tokens", 0L), 1, 0.0);
            }
        };
        Judgment record = custom.judge(exampleMessages(), exampleTools(), seoul());
        assertEquals(Boolean.TRUE, record.accepted());
        assertEquals(0.8, record.argumentsProbability());
    }
}
