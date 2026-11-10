package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static io.github.gogoymh.dmcall.Support.assertJson;
import static io.github.gogoymh.dmcall.Support.assertSameText;
import static io.github.gogoymh.dmcall.Support.conformance;
import static io.github.gogoymh.dmcall.Support.listedHeaders;
import static io.github.gogoymh.dmcall.Support.only;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

/** Every case of every file in {@code ../spec/conformance}, and {@code ../spec/prompts.json}. */
class ConformanceTest {
    @BeforeEach
    void isolate() {
        Support.isolate();
    }

    private static Stream<JsonNode> each(JsonNode array) {
        List<JsonNode> items = new ArrayList<>();
        array.forEach(items::add);
        return items.stream();
    }

    // prompts.json --------------------------------------------------------------------------
    @Test
    void promptsFileMatchesTheLibrary() {
        JsonNode data = Support.spec("prompts.json");
        assertEquals(data.get("system_prompt").textValue(), Prompts.SYSTEM_PROMPT);
        assertEquals(data.get("history_header").textValue(), Prompts.HISTORY_HEADER);
        assertEquals(data.get("history_footer").textValue(), Prompts.HISTORY_FOOTER);
        assertSameText(data.get("function_question"), Json.node(Prompts.FUNCTION_QUESTION), "function question");
        assertSameText(data.get("argument_question"), Json.node(Prompts.ARGUMENT_QUESTION), "argument question");
        assertEquals(data.get("repair_instruction").textValue(), Prompts.REPAIR_INSTRUCTION);
        assertJson(data.get("labels"), Json.node(Feedback.LABELS), "labels");
        assertJson(data.get("cut_points"), Json.node(Feedback.CUT_POINTS), "cut points");
        assertJson(data.get("label_fields"), Json.node(Feedback.LABEL_FIELDS), "label fields");
        List<String> thresholds = new ArrayList<>(Feedback.THRESHOLDS);
        thresholds.sort(String::compareTo);
        assertJson(data.get("threshold_fields"), Json.node(thresholds), "threshold fields");
        JsonNode defaults = data.get("defaults");
        assertEquals(defaults.get("threshold").doubleValue(), DMCall.DEFAULT_THRESHOLD);
        assertEquals(defaults.get("max_repairs").intValue(), DMCall.DEFAULT_MAX_REPAIRS);
        DMCall agent = DMCall.builder(new Support.ScriptedModel()).decisionModel(new Support.ScriptedJev().jev).build();
        assertEquals(defaults.get("checks").textValue(), agent.checks().value());
        assertEquals(defaults.get("history").textValue(), agent.history().value());
        assertEquals(defaults.get("feedback").textValue(), agent.feedback().value());
        assertEquals(defaults.get("on_exhaustion").textValue(), agent.onExhaustion().value());
        assertEquals(defaults.get("preserve_initial_no_call").booleanValue(), agent.preserveInitialNoCall());
        assertEquals(defaults.get("execute_unverified").booleanValue(), agent.executeUnverified());
        assertEquals(defaults.get("max_output_tokens").intValue(), OpenAI.builder("m").maxOutputTokens);
        JsonNode jev = data.get("jev");
        assertEquals(jev.get("model").textValue(), Jev.DEFAULT_MODEL);
        assertEquals(jev.get("base_url").textValue(), Jev.DEFAULT_BASE_URL);
        assertEquals(jev.get("feedback_source").textValue(), Jev.FEEDBACK_SOURCE);
        JsonNode pplx = data.get("pplx_decider");
        assertEquals(pplx.get("model_id").textValue(), PPLXDecider.MODEL_ID);
        assertEquals(pplx.get("revision").textValue(), PPLXDecider.MODEL_REVISION);
        assertEquals(pplx.get("calibration_temperature").doubleValue(), PPLXDecider.CALIBRATION_TEMPERATURE);
        assertEquals(pplx.get("max_input_tokens").intValue(), PPLXDecider.MAX_INPUT_TOKENS);
        assertEquals(pplx.get("server_model").textValue(), PPLXDecider.SERVER_MODEL);
        assertEquals(pplx.get("wire_model").textValue(), PPLXDecider.WIRE_MODEL);
        assertEquals(pplx.get("feedback_source").textValue(), PPLXDecider.FEEDBACK_SOURCE);
        PPLXDecider decider = PPLXDecider.builder("http://127.0.0.1:1").build();
        assertEquals(pplx.get("max_retries").intValue(), decider.maxRetries());
        assertEquals(pplx.get("timeout_s").doubleValue(), decider.timeout().toMillis() / 1000.0);
        List<Integer> statuses = new ArrayList<>(SystemOneDecisionModel.RETRY_STATUSES);
        statuses.sort(Integer::compare);
        assertJson(data.get("retry_statuses"), Json.node(statuses), "retry statuses");
        assertJson(data.get("incomplete_reasons"), Json.node(Provider.INCOMPLETE), "incomplete reasons");
    }

    // json_text.json ------------------------------------------------------------------------
    @TestFactory
    Stream<DynamicTest> jsonText() {
        return each(conformance("json_text.json").get("cases")).map(c -> dynamicTest(c.get("input").textValue(),
                () -> assertEquals(c.get("output").textValue(), Json.dumps(Json.parse(c.get("input").textValue())))));
    }

    // schema_check.json ---------------------------------------------------------------------
    @TestFactory
    Stream<DynamicTest> schemaCheck() {
        return each(conformance("schema_check.json").get("cases")).map(c -> dynamicTest(c.get("name").textValue(),
                () -> assertSameText(c.get("expected"), SchemaCheck.check(c.get("tools"), c.get("calls")).toJson(),
                        c.get("name").textValue())));
    }

    @TestFactory
    Stream<DynamicTest> schemaConfigurationErrors() {
        return each(conformance("schema_check.json").get("configuration_errors")).map(c -> dynamicTest(
                c.get("name").textValue(), () -> assertThrows(IllegalArgumentException.class,
                        () -> SchemaCheck.check(c.get("tools"), Json.array()))));
    }

    // tools.json ----------------------------------------------------------------------------
    @TestFactory
    Stream<DynamicTest> toolDefinitions() {
        return each(conformance("tools.json").get("cases")).map(c -> dynamicTest(c.get("name").textValue(), () -> {
            JsonNode before = c.get("input").deepCopy();
            assertSameText(c.get("expected"), Tool.of(c.get("input")).definition().get("function"), "definition");
            assertSameText(before, c.get("input"), "input after normalization");
        }));
    }

    @TestFactory
    Stream<DynamicTest> toolConfigurationErrors() {
        return each(conformance("tools.json").get("configuration_errors")).map(c -> dynamicTest(
                c.get("name").textValue(), () -> assertThrows(IllegalArgumentException.class,
                        () -> Tool.of(c.get("input")))));
    }

    // feedback.json -------------------------------------------------------------------------
    @TestFactory
    Stream<DynamicTest> feedback() {
        JsonNode data = conformance("feedback.json");
        List<DynamicTest> tests = new ArrayList<>();
        data.get("labels").forEach(c -> tests.add(dynamicTest("label " + c.get("p"), () -> {
            String label = Feedback.label(c.get("p"));
            assertEquals(c.get("label").isNull() ? null : c.get("label").textValue(), label);
        })));
        data.get("invalid_probabilities").forEach(bad -> tests.add(dynamicTest("invalid " + bad,
                () -> assertThrows(IllegalArgumentException.class, () -> Feedback.label(bad)))));
        int[] index = {0};
        data.get("represent").forEach(c -> tests.add(dynamicTest("represent " + index[0]++ + " "
                + c.get("representation").textValue(), () -> {
            JsonNode before = c.get("feedback").deepCopy();
            assertSameText(c.get("expected"), Feedback.represent(c.get("feedback"),
                    c.get("representation").textValue()), "representation");
            assertSameText(before, c.get("feedback"), "input after representing");
        })));
        data.get("single_check").forEach(c -> tests.add(dynamicTest("single check " + c.get("keep").textValue(),
                () -> assertSameText(c.get("expected"), Feedback.singleCheck(c.get("feedback"),
                        c.get("keep").textValue()), "single check"))));
        for (JsonNode conflict : data.get("conflicts").get("feedback")) {
            for (JsonNode representation : data.get("conflicts").get("representations")) {
                tests.add(dynamicTest("conflict " + representation.textValue() + " " + conflict,
                        () -> assertThrows(IllegalArgumentException.class,
                                () -> Feedback.represent(conflict, representation.textValue()))));
            }
        }
        return tests.stream();
    }

    // normalize_calls.json ------------------------------------------------------------------
    @TestFactory
    Stream<DynamicTest> normalizeCalls() {
        int[] index = {0};
        return each(conformance("normalize_calls.json").get("cases")).map(c -> dynamicTest("case " + index[0]++,
                () -> assertSameText(c.get("expected"), Calls.toJson(Calls.normalize(c.get("input"))), "calls")));
    }

    // providers.json ------------------------------------------------------------------------
    private static final Map<String, Function<Support.Mock, Provider>> REQUEST_CASES = Map.of(
            "openai with history", t -> OpenAI.builder("gpt-5-nano").apiKey("KEY").reasoningEffort("none")
                    .transport(t).build(),
            "openai no tools", t -> OpenAI.builder("gpt-5-nano").apiKey("KEY").transport(t).build(),
            "openai-compatible local", t -> OpenAICompatible.builder("gemma-4-31b-it")
                    .baseUrl("http://localhost:8000/v1").transport(t).build(),
            "xai", t -> XAI.builder("grok-4").apiKey("KEY").transport(t).build(),
            "openrouter", t -> OpenRouter.builder("meta/llama").apiKey("KEY").temperature(0.2).transport(t).build(),
            "meta muse", t -> Meta.builder("muse-spark-1.3").apiKey("KEY").transport(t).build(),
            "azure", t -> AzureOpenAI.builder("my deployment").endpoint("https://res.openai.azure.com/").apiKey("KEY")
                    .transport(t).build(),
            "anthropic with history", t -> Anthropic.builder("claude-sonnet-5-5").apiKey("KEY").transport(t).build(),
            "gemini", t -> Gemini.builder("models/gemini-2.5-flash").apiKey("KEY").temperature(0).thinkingBudget(0)
                    .transport(t).build());
    private static final List<String> WITH_HISTORY = List.of("openai with history", "meta muse", "azure",
            "anthropic with history", "gemini");

    @TestFactory
    Stream<DynamicTest> providerRequests() {
        JsonNode data = conformance("providers.json");
        return each(data.get("requests")).map(c -> dynamicTest(c.get("name").textValue(), () -> {
            String name = c.get("name").textValue();
            Support.Mock mock = Support.Mock.always(HttpTransport.Response.json(200, Json.object()));
            assertTrue(REQUEST_CASES.containsKey(name), "No provider is built for the request case " + name);
            Provider provider = REQUEST_CASES.get(name).apply(mock);
            ArrayNode messages = (ArrayNode) data.get("example").get("messages").deepCopy();
            ArrayNode tools = (ArrayNode) data.get("example").get("tools").deepCopy();
            if (name.equals("openai no tools")) {
                messages = (ArrayNode) Support.json("[{\"role\": \"user\", \"content\": \"hi\"}]");
                tools = Json.array();
            }
            ArrayNode history = WITH_HISTORY.contains(name) ? (ArrayNode) data.get("history").deepCopy() : null;
            ArrayNode m = messages;
            ArrayNode t = tools;
            // An empty response body is not a valid answer; only the request is compared here.
            assertThrows(ProviderException.class, () -> provider.generate(m, t, history));
            HttpTransport.Request request = mock.requests.get(0);
            assertEquals(c.get("url").textValue(), request.url());
            assertJson(c.get("headers"), listedHeaders(request.headers()), name + " headers");
            assertSameText(c.get("body"), request.json(), name + " body");
            assertSameBytes(c.get("body"), request, name);
        }));
    }

    @TestFactory
    Stream<DynamicTest> providerDecoding() {
        JsonNode data = conformance("providers.json");
        Map<String, Function<Support.Mock, Provider>> classes = Map.of(
                "openai", t -> OpenAI.builder("m").apiKey("KEY").transport(t).build(),
                "anthropic", t -> Anthropic.builder("m").apiKey("KEY").transport(t).build(),
                "gemini", t -> Gemini.builder("m").apiKey("KEY").transport(t).build());
        return each(data.get("decode")).map(c -> dynamicTest(c.get("name").textValue(), () -> {
            Support.Mock mock = Support.Mock.serving(c.get("response"));
            Provider provider = classes.get(c.get("provider").textValue()).apply(mock);
            Proposal proposal = provider.generate((ArrayNode) data.get("example").get("messages"),
                    (ArrayNode) data.get("example").get("tools"), null);
            assertSameText(c.get("expected"), only(proposal.toJson(), c.get("expected")), c.get("name").textValue());
        }));
    }

    @Test
    void filteredPromptIsAnUnreadableCandidate() {
        JsonNode data = conformance("providers.json");
        JsonNode filtered = data.get("filtered_prompt");
        Support.Mock mock = Support.Mock.replies(HttpTransport.Response.json(filtered.get("status").intValue(),
                filtered.get("response")));
        Proposal proposal = OpenAI.builder("m").apiKey("KEY").transport(mock).build()
                .generate((ArrayNode) data.get("example").get("messages"), (ArrayNode) data.get("example").get("tools"),
                        null);
        assertEquals(filtered.get("expected_incomplete").textValue(), proposal.incomplete());
        assertEquals(null, proposal.calls());
        assertTrue(proposal.usage().isEmpty());
    }

    // loop_edges.json -----------------------------------------------------------------------
    static Provider generator(JsonNode spec, HttpTransport transport) {
        String kind = spec.get("kind").textValue();
        String model = spec.get("model").textValue();
        JsonNode options = spec.has("options") ? spec.get("options") : Json.object();
        Provider.Builder<?, ?> builder;
        switch (kind) {
            case "openai":
                builder = OpenAI.builder(model);
                break;
            case "openai-compatible":
                builder = OpenAICompatible.builder(model);
                break;
            case "azure":
                builder = AzureOpenAI.builder(model).endpoint(options.get("endpoint").textValue());
                break;
            case "xai":
                builder = XAI.builder(model);
                break;
            case "anthropic":
                builder = Anthropic.builder(model);
                break;
            case "gemini":
                builder = Gemini.builder(model);
                break;
            default:
                throw new AssertionError(kind);
        }
        for (Iterator<Map.Entry<String, JsonNode>> it = options.fields(); it.hasNext(); ) {
            Map.Entry<String, JsonNode> option = it.next();
            JsonNode value = option.getValue();
            switch (option.getKey()) {
                case "api_key":
                    builder.apiKey(value.textValue());
                    break;
                case "base_url":
                    builder.baseUrl(value.textValue());
                    break;
                case "temperature":
                    builder.temperature(value.numberValue());
                    break;
                case "reasoning_effort":
                    ((ChatCompletionsProvider.ChatBuilder<?, ?>) builder).reasoningEffort(value.textValue());
                    break;
                case "endpoint":
                    break;
                default:
                    throw new AssertionError("Unmapped generator option " + option.getKey());
            }
        }
        builder.transport(transport).sleeper(Support.NO_SLEEP);
        return builder.build();
    }

    static DMCall.Builder configure(DMCall.Builder builder, JsonNode options) {
        for (Iterator<Map.Entry<String, JsonNode>> it = options.fields(); it.hasNext(); ) {
            Map.Entry<String, JsonNode> option = it.next();
            JsonNode value = option.getValue();
            switch (option.getKey()) {
                case "feedback":
                    builder.feedback(value.textValue());
                    break;
                case "history":
                    builder.history(value.textValue());
                    break;
                case "checks":
                    builder.checks(value.textValue());
                    break;
                case "max_repairs":
                    builder.maxRepairs(value.intValue());
                    break;
                case "on_exhaustion":
                    builder.onExhaustion(value.textValue());
                    break;
                case "preserve_initial_no_call":
                    builder.preserveInitialNoCall(value.booleanValue());
                    break;
                default:
                    throw new AssertionError("Unmapped DMCall option " + option.getKey());
            }
        }
        return builder;
    }

    @TestFactory
    Stream<DynamicTest> loopEdges() {
        JsonNode data = conformance("loop_edges.json");
        List<DynamicTest> tests = new ArrayList<>();
        for (Iterator<Map.Entry<String, JsonNode>> it = data.get("scenarios").fields(); it.hasNext(); ) {
            Map.Entry<String, JsonNode> entry = it.next();
            JsonNode scenario = entry.getValue();
            tests.add(dynamicTest(entry.getKey(), () -> {
                Support.Mock g = Support.Mock.servingAll(scenario.get("generator_responses"));
                Support.Mock d = Support.Mock.servingAll(scenario.get("decision_responses"));
                Provider generator = generator(scenario.get("generator"), g);
                JsonNode dm = scenario.get("decision_model");
                DecisionModel decision = dm.get("kind").textValue().equals("pplx")
                        ? PPLXDecider.builder(dm.get("base_url").textValue()).transport(d).build()
                        : Jev.builder().apiKey("JEV-KEY").transport(d).build();
                DMCall agent = configure(DMCall.builder(generator).decisionModel(decision).tools(data.get("tools")),
                        scenario.get("options")).build();
                Result result = agent.call(data.get("messages").deepCopy());
                compareRequests(scenario.get("generator_requests"), g, "generator request");
                compareRequests(scenario.get("decision_requests"), d, "decision request");
                assertSameText(scenario.get("result"), result.toJson(), "result");
            }));
        }
        return tests.stream();
    }

    private static void compareRequests(JsonNode expected, Support.Mock mock, String what) {
        assertEquals(expected.size(), mock.requests.size(), what + " count");
        for (int i = 0; i < expected.size(); i++) {
            JsonNode recorded = expected.get(i);
            HttpTransport.Request sent = mock.requests.get(i);
            assertEquals(recorded.get("url").textValue(), sent.url(), what + " " + i + " url");
            assertJson(recorded.get("headers"), listedHeaders(sent.headers()), what + " " + i + " headers");
            assertSameText(recorded.get("body"), sent.json(), what + " " + i + " body");
            assertSameBytes(recorded.get("body"), sent, what + " " + i);
        }
    }

    /** The raw body equals httpx's encoding of the recorded body: json.dumps(..., separators=(",", ":")). */
    private static void assertSameBytes(JsonNode recorded, HttpTransport.Request sent, String what) {
        assertEquals(Json.compact(recorded), sent.bodyText(), what + " bytes");
    }

    // paper_parity.json ---------------------------------------------------------------------
    @TestFactory
    Stream<DynamicTest> paperParity() {
        JsonNode golden = conformance("paper_parity.json");
        String generatorKey = golden.get("generator_api_key").textValue();
        String jevKey = golden.get("jev_api_key").textValue();
        List<DynamicTest> tests = new ArrayList<>();
        for (Iterator<Map.Entry<String, JsonNode>> it = golden.get("scenarios").fields(); it.hasNext(); ) {
            Map.Entry<String, JsonNode> entry = it.next();
            JsonNode scenario = entry.getValue();
            tests.add(dynamicTest(entry.getKey(), () -> {
                Support.Mock g = Support.Mock.servingAll(scenario.get("generator_responses"));
                Support.Mock d = Support.Mock.servingAll(scenario.get("decision_responses"));
                Jev jev = Jev.builder().apiKey(jevKey).transport(d).build();
                Provider model = scenario.get("provider").textValue().equals("anthropic")
                        ? Anthropic.builder("claude-sonnet-5-5").apiKey(generatorKey).transport(g).build()
                        : OpenAI.builder("gpt-5-nano").apiKey(generatorKey).reasoningEffort("none").transport(g).build();
                DMCall agent = configure(DMCall.builder(model).decisionModel(jev).tools(golden.get("tools")),
                        scenario.get("options")).build();
                Result result = agent.call(golden.get("messages").deepCopy());
                assertBodies(scenario.get("generator_requests"), g, "generator request");
                assertBodies(scenario.get("decision_requests"), d, "decision request");
                assertSameText(scenario.get("result"), only(result.toJson(), scenario.get("result")), "result");
            }));
        }
        return tests.stream();
    }

    private static void assertBodies(JsonNode expected, Support.Mock mock, String what) {
        assertEquals(expected.size(), mock.requests.size(), what + " count");
        for (int i = 0; i < expected.size(); i++) {
            // The history message is sent as text, so its exact characters matter, not only its JSON.
            assertSameText(expected.get(i), mock.body(i), what + " " + i);
            assertSameBytes(expected.get(i), mock.requests.get(i), what + " " + i);
        }
    }

    @Test
    void verifyMatchesThePapersCode() {
        JsonNode golden = conformance("paper_parity.json");
        JsonNode recorded = golden.get("verify");
        Support.Mock d = Support.Mock.servingAll(recorded.get("decision_responses"));
        Jev jev = Jev.builder().apiKey(golden.get("jev_api_key").textValue()).transport(d).build();
        Supplier<List<Verdict>> verdicts = () -> List.of(
                Verifier.verify(recorded.get("messages"), recorded.get("calls").get(0), golden.get("tools"), jev),
                Verifier.verify(recorded.get("second_messages").textValue(), recorded.get("calls").get(1),
                        golden.get("tools"), jev));
        ArrayNode actual = Json.array();
        verdicts.get().forEach(verdict -> actual.add(verdict.toJson()));
        assertBodies(recorded.get("decision_requests"), d, "decision request");
        assertSameText(recorded.get("verdicts"), actual, "verdicts");
    }

    @Test
    void everyConformanceFileIsCovered() {
        List<String> files = new ArrayList<>();
        try (Stream<java.nio.file.Path> paths = java.nio.file.Files.list(Support.SPEC.resolve("conformance"))) {
            paths.forEach(path -> files.add(path.getFileName().toString()));
        } catch (java.io.IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
        files.sort(String::compareTo);
        assertEquals(List.of("feedback.json", "json_text.json", "loop_edges.json", "normalize_calls.json",
                "paper_parity.json", "providers.json", "schema_check.json", "tools.json"), files);
        // Every case list is replayed in full; only guard against a file that lost its cases.
        assertTrue(conformance("paper_parity.json").get("scenarios").size() > 0);
        assertTrue(conformance("loop_edges.json").get("scenarios").size() > 0);
        assertTrue(conformance("json_text.json").get("cases").size() > 0);
        assertTrue(conformance("schema_check.json").get("cases").size() > 0);
    }
}
