package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

import static io.github.gogoymh.dmcall.Support.ScriptedJev;
import static io.github.gogoymh.dmcall.Support.ScriptedModel;
import static io.github.gogoymh.dmcall.Support.busan;
import static io.github.gogoymh.dmcall.Support.seoul;
import static io.github.gogoymh.dmcall.Support.weather;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The DM-Call loop: acceptance, cumulative repair, the paper's variants, fallback and execution (test_loop.py). */
class LoopTest {
    @BeforeEach
    void isolate() {
        Support.isolate();
    }

    private static Tool weatherTool(ToolHandler handler) {
        return Tool.of(weather(), handler);
    }

    private static DMCall.Builder agent(ScriptedModel model, ScriptedJev decision) {
        return DMCall.builder(model).decisionModel(decision.jev).tools(weather());
    }

    @Test
    void repairsWithCumulativeFeedbackThenRuns() {
        List<String> ran = new ArrayList<>();
        ScriptedModel model = new ScriptedModel(busan(), List.of(), seoul());
        ScriptedJev decision = new ScriptedJev();
        DMCall loop = DMCall.builder(model).decisionModel(decision.jev).tools(weatherTool(arguments -> {
            ran.add(arguments.get("city").asText());
            return Map.of("city", arguments.get("city").asText(), "temp", 21);
        })).build();
        Result result = loop.run("Weather in Seoul?");
        assertEquals(seoul(), result.calls());
        assertEquals(Boolean.TRUE, result.accepted());
        assertEquals(2, result.repairs());
        assertFalse(result.exhausted());
        assertEquals(List.of("Seoul"), ran);
        assertTrue(result.executed());
        assertEquals(Map.of("city", "Seoul", "temp", 21), result.output());
        assertEquals(List.of(0, 1, 2), model.histories.stream().map(JsonNode::size).toList());
        assertEquals(Calls.toJson(busan()), model.histories.get(2).get(0).get("calls"));
        assertEquals(Json.array(), model.histories.get(2).get(1).get("calls"));
        assertEquals("arguments", model.histories.get(2).get(0).get("feedback").get("failed_checks").get(0).asText());
        // Hybrid feedback: the generator reads the score and its label; the trace keeps the score.
        JsonNode shown = model.histories.get(1).get(0).get("feedback").get("checks").get("arguments");
        assertEquals(0.05, shown.get("probability_yes").doubleValue());
        assertEquals("very likely incorrect", shown.get("feedback_label").textValue());
        assertFalse(result.trace().get(0).feedback().get("checks").get("arguments").has("feedback_label"));
        assertEquals(List.of(false, false, true), result.attempts().stream().map(Attempt::accepted).toList());
        assertNull(result.attempts().get(1).argumentsProbability());  // a no call skips the argument question
        assertEquals(Map.of("input_tokens", 30L, "output_tokens", 6L), result.usage().generator());
        assertEquals(Map.of("input_tokens", 900L, "output_tokens", 0L), result.usage().decisionModel());
        assertEquals(new Probabilities(0.95, 0.95), result.probabilities());
        assertTrue(result.toString().startsWith("Result(call=get_weather(city=\"Seoul\"), accepted=true, repairs=2"),
                result.toString());
    }

    @Test
    void rejectedCallsDoNotRunUnlessAsked() {
        List<String> ran = new ArrayList<>();
        Tool tool = weatherTool(arguments -> ran.add(arguments.get("city").asText()));
        DMCall loop = DMCall.builder(new ScriptedModel(busan(), busan(), busan(), busan()))
                .decisionModel(new ScriptedJev().jev).tools(tool).build();
        Result result = loop.run("Weather in Seoul?");
        assertEquals(Boolean.FALSE, result.accepted());
        assertTrue(result.exhausted());
        assertEquals(3, result.repairs());
        assertFalse(result.executed());
        assertTrue(ran.isEmpty());
        DMCall again = DMCall.builder(new ScriptedModel(busan(), busan())).decisionModel(new ScriptedJev().jev)
                .tools(tool).maxRepairs(1).build();
        result = again.run("Weather in Seoul?", CallOptions.create().executeUnverified(true));
        assertEquals(1, result.repairs());
        assertTrue(result.executed());
        assertEquals(List.of("Busan"), ran);
    }

    @Test
    void withoutAnAcceptedCandidateTheFirstOneIsFinal() {
        List<String> ran = new ArrayList<>();
        Tool tool = weatherTool(arguments -> {
            ran.add(arguments.get("city").asText());
            return arguments.get("city").asText();
        });
        DMCall first = DMCall.builder(new ScriptedModel(List.of(), busan())).decisionModel(new ScriptedJev().jev)
                .tools(tool).maxRepairs(1).preserveInitialNoCall(false).build();
        Result result = first.run("Weather in Seoul?", CallOptions.create().executeUnverified(true));
        assertTrue(result.exhausted());
        assertEquals(Boolean.FALSE, result.accepted());
        assertEquals(1, result.repairs());
        assertTrue(result.noCall());
        assertEquals(0, result.finalAttempt());
        assertFalse(result.executed());
        assertTrue(ran.isEmpty());
        assertEquals("No tool fits.", result.text());
        assertEquals(new Probabilities(0.05, null), result.probabilities());
        DMCall last = DMCall.builder(new ScriptedModel(List.of(), busan())).decisionModel(new ScriptedJev().jev)
                .tools(tool).maxRepairs(1).onExhaustion(OnExhaustion.LAST).preserveInitialNoCall(false).build();
        result = last.run("Weather in Seoul?", CallOptions.create().executeUnverified(true));
        assertEquals(busan(), result.calls());
        assertEquals(1, result.finalAttempt());
        assertTrue(result.executed());
        assertEquals(List.of("Busan"), ran);
        DMCall perCall = agent(new ScriptedModel(busan(), List.of(), busan(), seoul()), new ScriptedJev())
                .maxRepairs(1).build();
        assertEquals(List.of(), perCall.call("Weather in Seoul?", CallOptions.create().onExhaustion("last")).calls());
        Result passed = perCall.call("Weather in Seoul?");
        assertEquals(seoul(), passed.calls());
        assertEquals(Boolean.TRUE, passed.accepted());
        assertEquals(1, passed.finalAttempt());
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> perCall.call("x", CallOptions.create().onExhaustion("best")));
        assertTrue(error.getMessage().contains("on_exhaustion"));
    }

    @Test
    void unreadableCandidateIsRepairedAndNeverRun() {
        ScriptedJev decision = new ScriptedJev();
        DMCall loop = agent(new ScriptedModel(null, seoul()), decision)
                .function("get_weather", arguments -> arguments.get("city").asText()).build();
        Result result = loop.run("Weather in Seoul?");
        assertNull(result.attempts().get(0).calls());
        assertFalse(result.attempts().get(0).schemaErrors().isEmpty());
        assertEquals(Boolean.TRUE, result.accepted());
        assertEquals("Seoul", result.output());
        assertEquals(1, decision.requests.size());
        DMCall stuck = agent(new ScriptedModel((List<Call>) null), new ScriptedJev()).maxRepairs(0)
                .function("get_weather", arguments -> arguments.get("city").asText()).build();
        result = stuck.run("Weather?", CallOptions.create().executeUnverified(true));
        assertNull(result.calls());
        assertTrue(result.unreadable());
        assertFalse(result.executed());
        assertTrue(result.toString().startsWith("Result(unreadable candidate"));
    }

    @Test
    void initialNoCallIsReturnedWithoutVerification() {
        ScriptedModel model = new ScriptedModel(List.of());
        ScriptedJev decision = new ScriptedJev("Seoul", 0.95, 0.97);
        Result result = agent(model, decision).function("get_weather", arguments -> null).build().run("Tell me a joke.");
        assertTrue(result.noCall());
        assertNull(result.accepted());
        assertTrue(result.initialNoCallPreserved());
        assertFalse(result.executed());
        assertEquals("No tool fits.", result.text());
        assertTrue(decision.requests.isEmpty());
        assertNull(result.attempts().get(0).accepted());
        assertEquals(1, model.histories.size());
        ScriptedJev checking = new ScriptedJev("Seoul", 0.95, 0.97);
        Result checked = agent(new ScriptedModel(List.of()), checking).preserveInitialNoCall(false).build()
                .call("Tell me a joke.");
        assertEquals(Boolean.TRUE, checked.accepted());
        assertFalse(checked.initialNoCallPreserved());
        assertEquals(1, checking.requests.size());
    }

    @Test
    void unreadableInitialOutputIsNotANoCall() {
        ScriptedJev decision = new ScriptedJev();
        Result result = agent(new ScriptedModel(null, seoul()), decision).build().call("Weather in Seoul?");
        assertFalse(result.initialNoCallPreserved());
        assertEquals(1, result.repairs());
        assertEquals(Boolean.TRUE, result.accepted());
    }

    @Test
    void latestHistoryKeepsTheRealRound() {
        ScriptedModel model = new ScriptedModel(busan(), List.of(), seoul());
        agent(model, new ScriptedJev()).history(History.LATEST).build().call("Weather in Seoul?");
        assertEquals(List.of(0, 1, 1), model.histories.stream().map(JsonNode::size).toList());
        assertEquals(2, model.histories.get(2).get(0).get("round").intValue());
        assertEquals(Json.array(), model.histories.get(2).get(0).get("calls"));
    }

    @Test
    void singleQuestionVariants() {
        Result result = agent(new ScriptedModel(busan()), new ScriptedJev()).checks(Checks.FUNCTION).build()
                .call("Weather in Seoul?");  // the wrong city passes when only the function is checked
        assertEquals(Boolean.TRUE, result.accepted());
        assertEquals(0, result.repairs());
        List<String> checks = new ArrayList<>();
        result.trace().get(0).feedback().get("checks").fieldNames().forEachRemaining(checks::add);
        assertEquals(List.of("function_selection"), checks);
        result = agent(new ScriptedModel(List.of(), seoul()), new ScriptedJev()).checks(Checks.ARGUMENTS)
                .preserveInitialNoCall(false).build().call("Weather in Seoul?");  // a no call has no arguments to reject
        assertEquals(Boolean.TRUE, result.accepted());
        assertTrue(result.noCall());
        assertEquals(0, result.repairs());
    }

    @Test
    void feedbackRepresentationsReachOnlyTheGenerator() {
        Object[][] cases = {{FeedbackMode.NUMERIC, "threshold"}, {FeedbackMode.TEXT, "feedback_label"},
                {FeedbackMode.HYBRID, "feedback_label"}};
        for (Object[] c : cases) {
            FeedbackMode representation = (FeedbackMode) c[0];
            ScriptedModel model = new ScriptedModel(busan(), seoul());
            Result result = agent(model, new ScriptedJev()).feedback(representation).build().call("Weather in Seoul?");
            JsonNode shown = model.histories.get(1).get(0).get("feedback").get("checks").get("arguments");
            assertTrue(shown.has((String) c[1]), representation.value());
            assertEquals(0.05, result.trace().get(0).feedback().get("checks").get("arguments").get("probability_yes")
                    .doubleValue());
            assertEquals(representation != FeedbackMode.TEXT, shown.has("probability_yes"));
        }
    }

    @Test
    void optionsAndDefaults() {
        DMCall loop = agent(new ScriptedModel(), new ScriptedJev()).build();
        assertEquals(3, loop.maxRepairs());
        assertEquals(Checks.BOTH, loop.checks());
        assertEquals(History.CUMULATIVE, loop.history());
        assertEquals(FeedbackMode.HYBRID, loop.feedback());
        assertEquals(OnExhaustion.FIRST, loop.onExhaustion());
        assertTrue(loop.preserveInitialNoCall());
        assertFalse(loop.executeUnverified());
        assertThrows(IllegalArgumentException.class, () -> agent(new ScriptedModel(), new ScriptedJev()).checks("neither"));
        assertThrows(IllegalArgumentException.class, () -> agent(new ScriptedModel(), new ScriptedJev()).history("none"));
        assertThrows(IllegalArgumentException.class, () -> agent(new ScriptedModel(), new ScriptedJev()).feedback("words"));
        assertThrows(IllegalArgumentException.class,
                () -> agent(new ScriptedModel(), new ScriptedJev()).onExhaustion("best"));
        assertThrows(IllegalArgumentException.class,
                () -> agent(new ScriptedModel(), new ScriptedJev()).maxRepairs(-1).build());
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> DMCall.builder(new ScriptedModel()).decisionModel(new ScriptedJev().jev).threshold(0.9).build());
        assertTrue(error.getMessage().contains("decision model itself"));
        error = assertThrows(IllegalArgumentException.class, () -> DMCall.builder(new ScriptedModel()).build());
        assertTrue(error.getMessage().contains("TYPESAFE_API_KEY"));
        Support.environment(Map.of("TYPESAFE_API_KEY", "jev-env", "OPENAI_API_KEY", "openai-env"));
        try (DMCall built = DMCall.builder("openai:gpt-5-nano").threshold(0.9).build()) {
            assertInstanceOf(OpenAI.class, built.generator());
            Jev jev = (Jev) built.decisionModel();
            assertEquals(0.9, jev.functionThreshold());
            assertEquals(0.9, jev.argumentsThreshold());
        }
        try (DMCall built = DMCall.builder("openai:gpt-5-nano").build()) {  // the paper's settings by default
            Jev jev = assertInstanceOf(Jev.class, built.decisionModel());
            assertEquals("jev-1.13.0", jev.model());
            assertEquals(0.5, jev.functionThreshold());
            assertEquals(0.5, jev.argumentsThreshold());
        }
    }

    @Test
    void perCallOverridesAndMissingHandlers() {
        ScriptedModel model = new ScriptedModel(busan(), seoul());
        ScriptedJev decision = new ScriptedJev();
        DMCall loop = agent(model, decision).build();
        Result result = loop.call("Weather in Seoul?", CallOptions.create().maxRepairs(0));
        assertEquals(0, result.repairs());
        assertEquals(Boolean.FALSE, result.accepted());
        assertTrue(result.exhausted());
        NoSuchElementException missing = assertThrows(NoSuchElementException.class,
                () -> loop.run("Weather in Seoul?", CallOptions.create().tools(weather())));
        assertTrue(missing.getMessage().contains("No handler"));
        assertEquals(1, model.histories.size());
        assertEquals(1, decision.requests.size());
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> loop.call("x", CallOptions.create().function("nope", arguments -> null)));
        assertTrue(error.getMessage().contains("not one of the tools"));
        assertThrows(IllegalArgumentException.class, () -> loop.call("x", CallOptions.create().maxRepairs(-1)));
    }

    @Test
    void overriddenToolDefinitionsKeepHandlersAndExplicitHandlersTakePrecedence() {
        ScriptedModel model = new ScriptedModel(seoul(), seoul());
        DMCall loop = agent(model, new ScriptedJev()).function("get_weather", arguments -> "registered").build();
        assertEquals("registered", loop.run("Weather in Seoul?", CallOptions.create().tools(weather())).output());
        assertEquals("replacement", loop.run("Weather in Seoul?", CallOptions.create().tools(weather())
                .function("get_weather", arguments -> "replacement")).output());
    }

    @Test
    void missingHandlerFailsBeforeGeneratingEvenForAnInitialNoCall() {
        ScriptedModel model = new ScriptedModel(List.of());
        ScriptedJev decision = new ScriptedJev();
        DMCall loop = agent(model, decision).build();
        assertThrows(NoSuchElementException.class, () -> loop.run("Tell me a joke."));
        assertTrue(model.histories.isEmpty());
        assertTrue(decision.requests.isEmpty());
    }

    @Test
    void checkedHandlerFailuresAreWrapped() {
        DMCall loop = agent(new ScriptedModel(seoul()), new ScriptedJev())
                .function("get_weather", arguments -> {
                    throw new java.io.IOException("offline");
                }).build();
        ToolExecutionException error = assertThrows(ToolExecutionException.class, () -> loop.run("Weather in Seoul?"));
        assertEquals("get_weather", error.tool());
        assertInstanceOf(java.io.IOException.class, error.getCause());
    }

    @Test
    void asyncWrappers() throws Exception {
        DMCall loop = agent(new ScriptedModel(seoul(), seoul()), new ScriptedJev())
                .function("get_weather", arguments -> arguments.get("city").asText().toUpperCase()).build();
        Result called = loop.callAsync("Weather in Seoul?").get();
        Result ran = loop.runAsync("Weather in Seoul?").get();
        Verdict verdict = loop.verifyAsync("Weather in Seoul?", seoul()).get();
        assertEquals(Boolean.TRUE, called.accepted());
        assertFalse(called.executed());
        assertEquals("SEOUL", ran.output());
        assertTrue(verdict.accepted());
    }

    @Test
    void messagesAreValidated() {
        DMCall loop = agent(new ScriptedModel(seoul(), seoul()), new ScriptedJev()).build();
        loop.call(List.of(Map.of("role", "user", "content", List.of(Map.of("type", "text", "text", "Weather"),
                Map.of("type", "text", "text", "in Seoul?")))));
        loop.call(List.of(Message.system("Be brief."), Message.user("Weather in Seoul?")));
        for (Object bad : new Object[]{List.of(), List.of(Map.of("role", "tool", "content", "{}")),
                List.of(Map.of("role", "user", "content", Map.of("x", 1))), 3}) {
            assertThrows(IllegalArgumentException.class, () -> loop.call(bad), String.valueOf(bad));
        }
    }

    @Test
    void verifyChecksCallsFromAnySdk() {
        ScriptedJev decision = new ScriptedJev();
        String[] shapes = {
            "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Seoul\"}}",
            "{\"id\": \"c1\", \"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"arguments\":"
                    + " \"{\\\"city\\\": \\\"Seoul\\\"}\"}}",
            "{\"type\": \"tool_use\", \"id\": \"t1\", \"name\": \"get_weather\", \"input\": {\"city\": \"Seoul\"}}",
            "{\"functionCall\": {\"name\": \"get_weather\", \"args\": {\"city\": \"Seoul\"}}}",
        };
        for (String shape : shapes) {
            Verdict verdict = Verifier.verify("Weather in Seoul?", List.of(Json.parse(shape)), List.of(weather()),
                    decision.jev);
            assertTrue(verdict.accepted(), shape);
            assertEquals(seoul(), verdict.calls());
            assertEquals(0.95, verdict.argumentsProbability());
        }
        assertFalse(Verifier.verify("Weather in Seoul?", busan(), List.of(weather()), decision.jev).accepted());
        int count = decision.requests.size();
        Verdict malformed = Verifier.verify("Weather?", Map.of("name", "get_weather", "arguments", "{oops"),
                List.of(weather()), decision.jev);
        assertFalse(malformed.accepted());
        assertEquals(count, decision.requests.size());  // rejected locally; nothing is sent
        assertEquals(List.of(), Verifier.verify("Tell me a joke.", null, List.of(weather()), decision.jev).calls());
        assertTrue(Verifier.verify("Tell me a joke.", List.of(), List.of(weather()),
                new ScriptedJev("Seoul", 0.95, 0.9).jev).accepted());

        ObjectNode sdkMessage = (ObjectNode) Json.parse("{\"role\": \"assistant\", \"content\": null, \"refusal\": null,"
                + " \"tool_calls\": [{\"id\": \"c0\", \"type\": \"function\", \"function\": {\"name\": \"get_weather\","
                + " \"arguments\": \"{\\\"city\\\": \\\"Busan\\\"}\"}}]}");
        List<Object> history = List.of(Map.of("role", "user", "content", "Weather in Busan, then Seoul?"), sdkMessage,
                Map.of("role", "tool", "tool_call_id", "c0", "content", "{\"temp\": 18}"));
        assertTrue(Verifier.verify(history, seoul(), List.of(weather()), decision.jev).accepted());
        JsonNode sent = decision.requests.get(decision.requests.size() - 1).get("state").get("messages");
        List<String> roles = new ArrayList<>();
        sent.forEach(message -> roles.add(message.get("role").asText()));
        assertEquals(List.of("user", "assistant", "tool"), roles);
        assertFalse(sent.get(1).has("refusal"));
        assertFalse(sent.get(1).has("content"));  // a null content is dropped
        assertThrows(IllegalArgumentException.class, () -> Verifier.verify(List.of(Map.of("role", "robot", "content",
                "x")), seoul(), List.of(weather()), decision.jev));
        DMCall loop = agent(new ScriptedModel(), decision).build();
        assertTrue(loop.verify("Weather in Seoul?", seoul()).accepted());
    }

    @Test
    void closeReachesOwnedClientsOnly() {
        ScriptedModel model = new ScriptedModel();
        DMCall loop = DMCall.builder(model).decisionModel(new ScriptedJev().jev).tools(weather()).build();
        loop.close();
        assertFalse(model.closed);  // a generator passed in belongs to the caller
    }

    @Test
    void customDecisionModelsPlugIn() {
        DecisionModel always = (messages, tools, calls) -> Judgment.builder().functionPass(true).argumentsPass(true)
                .accepted(true).functionProbability(1.0).argumentsProbability(1.0).feedback(Json.object()).build();
        Result result = DMCall.builder(new ScriptedModel(busan())).decisionModel(always).tools(weather()).build()
                .call("Weather in Seoul?");
        assertEquals(Boolean.TRUE, result.accepted());
        assertEquals(busan(), result.calls());
    }
}
