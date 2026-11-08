package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static io.github.gogoymh.dmcall.Support.json;
import static io.github.gogoymh.dmcall.Support.weather;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The schema check S(c) and the feedback representations (test_schema_and_feedback.py). */
class SchemaFeedbackTest {
    @BeforeEach
    void isolate() {
        Support.isolate();
    }

    private static SchemaReport check(Object arguments, Object tools, String name) {
        return SchemaCheck.check(tools, List.of(Map.of("name", name, "arguments", arguments)));
    }

    private static SchemaReport check(Object arguments) {
        return check(arguments, List.of(weather()), "get_weather");
    }

    // S(c) -----------------------------------------------------------------------------------
    @Test
    void validCallAndNoCallPass() {
        assertTrue(check(Map.of("city", "Seoul")).accepted());
        SchemaReport empty = SchemaCheck.check(List.of(weather()), List.of());
        assertTrue(empty.accepted());
        assertTrue(empty.wellFormed());
        assertTrue(empty.errors().isEmpty());
    }

    @Test
    void argumentStructureErrors() {
        for (Object arguments : List.of(Map.of(), Map.of("city", "Seoul", "invented", 1), Map.of("city", 3))) {
            SchemaReport result = check(arguments);
            assertTrue(result.functionPass());
            assertFalse(result.argumentsPass());
            assertTrue(result.wellFormed());
        }
    }

    @Test
    void unknownFunctionIsReadableButNotAvailable() {
        SchemaReport result = check(Map.of(), List.of(weather()), "invented");
        assertFalse(result.functionPass());
        assertFalse(result.argumentsPass());
        assertTrue(result.wellFormed());
    }

    @Test
    void unreadableCandidatesAreNotWellFormed() {
        ObjectNode nan = Json.object();
        nan.put("name", "get_weather");
        nan.putObject("arguments").put("city", Double.NaN);
        List<Object> candidates = new ArrayList<>();
        candidates.add(null);
        candidates.add(List.of("get_weather"));
        candidates.add(List.of(Map.of("name", "get_weather", "arguments", "{}")));
        candidates.add(List.of(nan));
        candidates.add(List.of(Map.of("name", "", "arguments", Map.of())));
        for (Object calls : candidates) {
            SchemaReport result = SchemaCheck.check(List.of(weather()), calls);
            assertFalse(result.accepted(), String.valueOf(calls));
            assertFalse(result.wellFormed(), String.valueOf(calls));
        }
    }

    @Test
    void typesAndRequiredNamesOnlyNotValueConstraints() {
        JsonNode tools = json("[{\"type\": \"function\", \"function\": {\"name\": \"book\", \"description\": \"\","
                + " \"parameters\": {\"type\": \"object\", \"properties\": {\"genre\": {\"type\": \"string\", \"enum\":"
                + " [\"Horror\", \"Comedy\"]}, \"rooms\": {\"type\": \"integer\", \"minimum\": 1}, \"metrics\": {\"type\":"
                + " \"array\", \"enum\": [\"view\", \"trust\"], \"items\": {\"type\": \"string\"}}, \"start\": {\"type\":"
                + " \"string\", \"default\": null}, \"end\": {\"type\": \"string\"}, \"filters\": {\"type\": \"object\","
                + " \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}, \"required\":"
                + " [\"genre\"]}}}]");
        // Values outside an enum or range are the decision model's question; a declared null default is admissible.
        SchemaReport passed = check(json("{\"genre\": \"Any\", \"rooms\": 0, \"metrics\": [\"view\"], \"start\": null,"
                + " \"filters\": {\"city\": \"Seoul\"}}"), tools, "book");
        assertTrue(passed.accepted());
        assertTrue(passed.errors().isEmpty());
        for (String arguments : new String[]{"{\"genre\": true}", "{\"genre\": \"Any\", \"rooms\": \"2\"}",
                "{\"genre\": \"Any\", \"end\": null}", "{\"genre\": \"Any\", \"metrics\": \"view\"}", "{\"rooms\": 1}",
                "{\"genre\": \"Any\", \"extra\": 1}", "{\"genre\": \"Any\", \"filters\": {}}"}) {
            SchemaReport result = check(json(arguments), tools, "book");
            assertTrue(result.functionPass(), arguments);
            assertFalse(result.argumentsPass(), arguments);
        }
    }

    @Test
    void alternativesAndOpenKeywords() {
        JsonNode tools = json("[{\"type\": \"function\", \"function\": {\"name\": \"f\", \"description\": \"\","
                + " \"parameters\": {\"type\": \"object\", \"additionalProperties\": false, \"required\": [\"x\"],"
                + " \"properties\": {\"x\": {\"oneOf\": [{\"type\": \"string\", \"enum\": [\"a\"]}, {\"type\": \"string\","
                + " \"maxLength\": 1}]}}}}}]");
        assertTrue(check(Map.of("x", "abc"), tools, "f").accepted());
        for (Object arguments : List.of(Map.of("x", 3), Map.of(), Map.of("x", "a", "y", 1))) {
            assertFalse(check(arguments, tools, "f").accepted(), String.valueOf(arguments));
        }
        ObjectNode open = weather();
        ((ObjectNode) open.get("function").get("parameters")).put("additionalProperties", true);
        assertTrue(check(Map.of("city", "Seoul", "extra", 1), List.of(open), "get_weather").accepted());
    }

    @Test
    void patternPropertiesAndClosedArrays() {
        JsonNode tools = json("[{\"type\": \"function\", \"function\": {\"name\": \"p\", \"description\": \"\","
                + " \"parameters\": {\"type\": \"object\", \"properties\": {\"pair\": {\"type\": \"array\","
                + " \"prefixItems\": [{\"type\": \"integer\"}], \"items\": false}}, \"patternProperties\": {\"^n_\":"
                + " {\"type\": \"number\"}}, \"additionalProperties\": false}}}]");
        SchemaReport result = check(json("{\"pair\": [1, 2, 3], \"n_a\": \"x\", \"z\": 1}"), tools, "p");
        List<String> messages = new ArrayList<>();
        result.errors().forEach(error -> messages.add(error.path() + " " + error.message()));
        assertEquals(List.of("calls[0].arguments[\"n_a\"] 'x' is not of type 'number'",
                "calls[0].arguments[\"pair\"] Expected at most 1 item but found 2 extra: [2, 3]",
                "calls[0].arguments 'z' does not match any of the regexes: '^n_'"), messages);
    }

    @Test
    void falseSubschemasReportAtTheParentPathLikeJsonschema() {
        JsonNode tools = json("[{\"type\": \"function\", \"function\": {\"name\": \"f\", \"description\": \"\","
                + " \"parameters\": {\"type\": \"object\", \"properties\": {\"f\": {\"type\": \"object\", \"properties\":"
                + " {\"x\": false}}, \"p\": {\"type\": \"array\", \"prefixItems\": [false]}, \"q\": {\"anyOf\": [false,"
                + " {\"type\": \"string\"}]}, \"r\": {\"$ref\": \"#/$defs/no\"}}, \"$defs\": {\"no\": false},"
                + " \"patternProperties\": {\"^z\": false}}}}]");
        SchemaReport result = check(json("{\"f\": {\"x\": 1}, \"p\": [2], \"q\": 3, \"r\": 4, \"zz\": 5}"), tools, "f");
        List<String> messages = new ArrayList<>();
        result.errors().forEach(error -> messages.add(error.path() + " " + error.message()));
        // Recorded from the Python reference.
        assertEquals(List.of("calls[0].arguments[\"f\"] False schema does not allow 1",
                "calls[0].arguments[\"p\"] False schema does not allow 2",
                "calls[0].arguments[\"q\"] 3 is not valid under any of the given schemas",
                "calls[0].arguments[\"r\"] False schema does not allow 4",
                "calls[0].arguments False schema does not allow 5"), messages);
    }

    @Test
    void referencesResolveWithinTheSchema() {
        JsonNode tools = json("[{\"type\": \"function\", \"function\": {\"name\": \"r\", \"description\": \"\","
                + " \"parameters\": {\"type\": \"object\", \"properties\": {\"a\": {\"$ref\": \"#/definitions/s\"},"
                + " \"b\": {\"$ref\": \"#/$defs/missing\"}}, \"definitions\": {\"s\": {\"type\": \"string\"}}}}}]");
        assertEquals("1 is not of type 'string'", check(Map.of("a", 1), tools, "r").errors().get(0).message());
        assertThrows(IllegalArgumentException.class, () -> check(Map.of("b", 1), tools, "r"));
    }

    @Test
    void nonAsciiPathsAreEscapedLikePython() {
        JsonNode tools = json("[{\"type\": \"function\", \"function\": {\"name\": \"k\", \"description\": \"\","
                + " \"parameters\": {\"type\": \"object\", \"properties\": {\"도시\": {\"type\": \"string\"}}}}}]");
        SchemaError error = check(Map.of("도시", 1), tools, "k").errors().get(0);
        assertEquals("calls[0].arguments[\"\\ub3c4\\uc2dc\"]", error.path());
    }

    @Test
    void invalidToolSchemaIsAConfigurationError() {
        JsonNode bad = json("[{\"type\": \"function\", \"function\": {\"name\": \"f\", \"parameters\": {\"type\":"
                + " \"object\", \"properties\": {\"a\": {\"type\": \"nonsense\"}}}}}]");
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> SchemaCheck.check(bad, List.of()));
        assertTrue(error.getMessage().contains("invalid JSON Schema"));
        error = assertThrows(IllegalArgumentException.class,
                () -> SchemaCheck.check(List.of(weather(), weather()), List.of()));
        assertTrue(error.getMessage().contains("unique"));
        for (String schema : new String[]{"{\"type\": \"object\", \"required\": \"city\"}",
                "{\"type\": \"object\", \"properties\": {\"a\": {\"anyOf\": []}}}",
                "{\"type\": \"object\", \"properties\": {\"a\": {\"items\": [{\"type\": \"string\"}]}}}",
                "{\"type\": \"object\", \"properties\": {\"a\": {\"minLength\": -1}}}",
                "{\"type\": \"object\", \"patternProperties\": {\"(\": {}}}"}) {
            JsonNode tools = json("[{\"name\": \"f\", \"parameters\": " + schema + "}]");
            assertThrows(IllegalArgumentException.class, () -> SchemaCheck.check(tools, List.of()), schema);
        }
    }

    // Feedback ---------------------------------------------------------------------------------
    @Test
    void labelBands() {
        Object[][] cases = {{0.0, "very likely incorrect"}, {0.19, "very likely incorrect"}, {0.2, "likely incorrect"},
                {0.48, "uncertain"}, {0.52, "uncertain"}, {0.6, "likely correct"}, {0.8, "very likely correct"},
                {1.0, "very likely correct"}, {null, null}};
        for (Object[] c : cases) {
            assertEquals(c[1], Feedback.label((Double) c[0]));
        }
    }

    @Test
    void labelRejectsNonProbabilities() {
        for (JsonNode bad : new JsonNode[]{json("-0.1"), json("1.1"), json("NaN"), json("true"), json("\"0.5\"")}) {
            assertThrows(IllegalArgumentException.class, () -> Feedback.label(bad), bad.toString());
        }
    }

    private static final JsonNode FEEDBACK = json("{\"source\": \"jev_noul_and_deterministic_schema_validation\","
            + " \"checks\": {\"function_selection\": {\"pass\": true, \"probability_yes\": 0.95, \"threshold\": 0.5,"
            + " \"schema_pass\": true}, \"arguments\": {\"skipped\": false, \"pass\": false, \"probability_yes\": 0.12,"
            + " \"threshold\": 0.5, \"schema_pass\": true}}, \"failed_checks\": [\"arguments\"], \"schema_errors\":"
            + " [{\"check\": \"arguments\", \"path\": \"calls[0].arguments\", \"message\": \"'probability_yes' 0.3\"}],"
            + " \"instruction\": \"Reconsider ... The probabilities are verifier judgments, not ground truth or"
            + " explanations.\"}");

    private static List<String> keys(JsonNode node) {
        List<String> keys = new ArrayList<>();
        node.fieldNames().forEachRemaining(keys::add);
        return keys;
    }

    @Test
    void hybridKeepsNumbersAndThresholdsAndAddsLabels() {
        ObjectNode out = Feedback.represent(FEEDBACK, FeedbackMode.HYBRID);
        JsonNode arguments = out.get("checks").get("arguments");
        assertEquals(List.of("skipped", "pass", "probability_yes", "feedback_label", "threshold", "schema_pass"),
                keys(arguments));
        assertEquals(0.12, arguments.get("probability_yes").doubleValue());
        assertEquals("very likely incorrect", arguments.get("feedback_label").textValue());
        assertEquals("very likely correct", out.get("checks").get("function_selection").get("feedback_label").textValue());
        assertEquals(FEEDBACK.get("schema_errors"), out.get("schema_errors"));
        assertEquals(FEEDBACK.get("instruction"), out.get("instruction"));
        assertFalse(FEEDBACK.get("checks").get("arguments").has("feedback_label"));  // the input is not changed
    }

    @Test
    void textReplacesNumbersAndThresholdsWithLabels() {
        ObjectNode out = Feedback.represent(FEEDBACK, FeedbackMode.TEXT);
        assertEquals(json("{\"skipped\": false, \"pass\": false, \"feedback_label\": \"very likely incorrect\","
                + " \"schema_pass\": true}"), out.get("checks").get("arguments"));
        assertTrue(out.get("instruction").textValue()
                .endsWith("The feedback labels are verifier judgments, not ground truth or explanations."));
        assertEquals(FEEDBACK.get("schema_errors"), out.get("schema_errors"));  // task diagnostics stay verbatim
    }

    @Test
    void numericIsUnchangedAndSkippedQuestionsStayNull() {
        assertEquals(FEEDBACK, Feedback.represent(FEEDBACK, FeedbackMode.NUMERIC));
        JsonNode skipped = json("{\"checks\": {\"arguments\": {\"skipped\": true, \"pass\": null, \"probability_yes\":"
                + " null, \"threshold\": 0.5}}}");
        assertTrue(Feedback.represent(skipped, FeedbackMode.HYBRID).get("checks").get("arguments").get("feedback_label")
                .isNull());
        assertEquals(json("{\"skipped\": true, \"pass\": null, \"feedback_label\": null}"),
                Feedback.represent(skipped, FeedbackMode.TEXT).get("checks").get("arguments"));
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> Feedback.represent(FEEDBACK, "words"));
        assertTrue(error.getMessage().contains("feedback"));
        ArrayNode history = Json.array();
        history.addObject().putArray("calls");
        ((ObjectNode) history.get(0)).set("feedback", FEEDBACK);
        assertEquals(Feedback.represent(FEEDBACK, FeedbackMode.TEXT),
                Feedback.representHistory(history, FeedbackMode.TEXT).get(0).get("feedback"));
        assertThrows(IllegalArgumentException.class,
                () -> Feedback.representHistory(json("[{\"calls\": []}]"), FeedbackMode.HYBRID));
    }

    @Test
    void singleCheckRemovesOnlyTheOtherQuestion() {
        ObjectNode functionOnly = Feedback.singleCheck(FEEDBACK, "function");
        assertEquals(List.of("function_selection"), keys(functionOnly.get("checks")));
        assertTrue(functionOnly.get("failed_checks").isEmpty());
        ObjectNode argumentsOnly = Feedback.singleCheck(FEEDBACK, Checks.ARGUMENTS);
        assertEquals(List.of("arguments"), keys(argumentsOnly.get("checks")));
        assertEquals(json("[\"arguments\"]"), argumentsOnly.get("failed_checks"));
        assertEquals(FEEDBACK.get("schema_errors"), argumentsOnly.get("schema_errors"));
        assertThrows(IllegalArgumentException.class, () -> Feedback.singleCheck(FEEDBACK, "both"));
        assertThrows(IllegalArgumentException.class, () -> Feedback.singleCheck(FEEDBACK, Checks.BOTH));
        assertNull(Feedback.label((Double) null));
    }
}
