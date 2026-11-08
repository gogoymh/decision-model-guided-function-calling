package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.fail;

/** Shared test helpers. Every test is offline: HTTP goes through {@link Mock}. */
final class Support {
    static final String SECRET = "sk-test-SECRET-123";
    static final Path SPEC = Path.of("..", "spec");

    private Support() {
    }

    // Environment -----------------------------------------------------------------------------
    /** Replaces the environment with {@code values} (and nothing else). */
    static Map<String, String> environment(Map<String, String> values) {
        Map<String, String> env = new HashMap<>(values);
        Env.source = env::get;
        return env;
    }

    static void isolate() {
        environment(Map.of());
    }

    // Spec files ------------------------------------------------------------------------------
    static JsonNode spec(String name) {
        try {
            return Json.parse(Files.readString(SPEC.resolve(name), StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    static JsonNode conformance(String name) {
        return spec("conformance/" + name);
    }

    static JsonNode json(String text) {
        return Json.parse(text);
    }

    // JSON comparison -------------------------------------------------------------------------
    /** Python's {@code ==} on JSON values, except that booleans never equal numbers. */
    static boolean pyEquals(JsonNode a, JsonNode b) {
        boolean noneA = a == null || a.isNull() || a.isMissingNode();
        boolean noneB = b == null || b.isNull() || b.isMissingNode();
        if (noneA || noneB) {
            return noneA && noneB;
        }
        if (a.isNumber() && b.isNumber()) {
            if (a.isFloatingPointNumber() && Double.isNaN(a.doubleValue())) {
                return false;
            }
            if (b.isFloatingPointNumber() && Double.isNaN(b.doubleValue())) {
                return false;
            }
            return a.decimalValue().compareTo(b.decimalValue()) == 0;
        }
        if (a.isBoolean() && b.isBoolean()) {
            return a.booleanValue() == b.booleanValue();
        }
        if (a.isTextual() && b.isTextual()) {
            return a.textValue().equals(b.textValue());
        }
        if (a.isArray() && b.isArray()) {
            if (a.size() != b.size()) {
                return false;
            }
            for (int i = 0; i < a.size(); i++) {
                if (!pyEquals(a.get(i), b.get(i))) {
                    return false;
                }
            }
            return true;
        }
        if (a.isObject() && b.isObject()) {
            if (a.size() != b.size()) {
                return false;
            }
            for (Iterator<Map.Entry<String, JsonNode>> it = a.fields(); it.hasNext(); ) {
                Map.Entry<String, JsonNode> entry = it.next();
                if (!b.has(entry.getKey()) || !pyEquals(entry.getValue(), b.get(entry.getKey()))) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    static void assertJson(JsonNode expected, JsonNode actual, String what) {
        if (!pyEquals(expected, actual)) {
            fail(what + " differs.\nexpected: " + Json.dumps(expected) + "\nactual:   " + Json.dumps(actual));
        }
    }

    /** Equal as JSON values and as Python {@code json.dumps} text (key order and number forms). */
    static void assertSameText(JsonNode expected, JsonNode actual, String what) {
        assertJson(expected, actual, what);
        String e = Json.dumps(expected);
        String a = Json.dumps(actual);
        if (!e.equals(a)) {
            fail(what + " serializes differently.\nexpected: " + e + "\nactual:   " + a);
        }
    }

    /** The keys of {@code full} that {@code subset} names. */
    static ObjectNode only(JsonNode full, JsonNode subset) {
        ObjectNode result = Json.object();
        subset.fieldNames().forEachRemaining(key -> result.set(key, full.get(key)));
        return result;
    }

    // HTTP ----------------------------------------------------------------------------------
    /** A transport that records every request and answers with a handler. */
    static final class Mock implements HttpTransport {
        final List<Request> requests = new ArrayList<>();
        private final Function<Request, Response> handler;
        boolean closed;

        Mock(Function<Request, Response> handler) {
            this.handler = handler;
        }

        /** Serves these JSON bodies with status 200, in order. */
        static Mock serving(JsonNode... bodies) {
            Deque<JsonNode> queue = new ArrayDeque<>(List.of(bodies));
            return new Mock(request -> {
                if (queue.isEmpty()) {
                    fail("Unexpected request to " + request.url());
                }
                return Response.json(200, queue.removeFirst());
            });
        }

        /** Serves the items of a JSON array, in order. */
        static Mock servingAll(JsonNode bodies) {
            List<JsonNode> list = new ArrayList<>();
            bodies.forEach(list::add);
            return serving(list.toArray(new JsonNode[0]));
        }

        /** Answers with these responses in order. */
        static Mock replies(Response... responses) {
            Deque<Response> queue = new ArrayDeque<>(List.of(responses));
            return new Mock(request -> {
                if (queue.isEmpty()) {
                    fail("Unexpected request to " + request.url());
                }
                return queue.removeFirst();
            });
        }

        static Mock always(Response response) {
            return new Mock(request -> response);
        }

        @Override
        public Response send(Request request) throws IOException {
            requests.add(request);
            Response response = handler.apply(request);
            if (response == null) {
                throw new java.net.ConnectException("connection refused");
            }
            return response;
        }

        @Override
        public void close() {
            closed = true;
        }

        JsonNode body(int index) {
            return requests.get(index).json();
        }

        List<JsonNode> bodies() {
            List<JsonNode> bodies = new ArrayList<>();
            requests.forEach(request -> bodies.add(request.json()));
            return bodies;
        }
    }

    static final Sleeper NO_SLEEP = seconds -> { };

    /** The headers the spec compares, by lower-case name. */
    static ObjectNode listedHeaders(Map<String, String> headers) {
        List<String> listed = List.of("authorization", "x-api-key", "api-key", "x-goog-api-key", "anthropic-version",
                "content-type");
        ObjectNode result = Json.object();
        headers.forEach((name, value) -> {
            if (listed.contains(name.toLowerCase(java.util.Locale.ROOT))) {
                result.put(name.toLowerCase(java.util.Locale.ROOT), value);
            }
        });
        return result;
    }

    // Fixtures (tests/conftest.py) ----------------------------------------------------------
    static ObjectNode weather() {
        return (ObjectNode) json("{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\":"
                + " \"Current weather for a city.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\":"
                + " {\"type\": \"string\"}, \"unit\": {\"type\": \"string\", \"enum\": [\"celsius\", \"fahrenheit\"]}},"
                + " \"required\": [\"city\"]}}}");
    }

    static ArrayNode exampleMessages() {
        return (ArrayNode) json("[{\"role\": \"system\", \"content\": \"Be brief.\"}, {\"role\": \"user\", \"content\":"
                + " \"Weather in Seoul?\"}]");
    }

    static ArrayNode exampleTools() {
        ArrayNode tools = Json.array();
        tools.add(weather());
        return tools;
    }

    static ArrayNode history() {
        return (ArrayNode) json("[{\"calls\": [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Busan\"}}],"
                + " \"feedback\": {\"failed_checks\": [\"arguments\"], \"instruction\": \"Reconsider.\"}},"
                + " {\"round\": 2, \"calls\": [], \"feedback\": {\"failed_checks\": [\"function_selection\"]}}]");
    }

    static List<Call> seoul() {
        return List.of(Call.of("get_weather", Map.of("city", "Seoul")));
    }

    static List<Call> busan() {
        return List.of(Call.of("get_weather", Map.of("city", "Busan")));
    }

    /** A Chat Completions response body. */
    static ObjectNode chat(List<String[]> calls, String content, String finish, ObjectNode usage,
                           Map<String, Object> message) {
        ObjectNode body = Json.object();
        body.put("model", "resolved-model");
        ObjectNode choice = body.putArray("choices").addObject();
        choice.put("finish_reason", finish != null ? finish : calls.isEmpty() ? "stop" : "tool_calls");
        ObjectNode m = choice.putObject("message");
        m.put("role", "assistant");
        m.put("content", content);
        if (!calls.isEmpty()) {
            ArrayNode toolCalls = m.putArray("tool_calls");
            for (int i = 0; i < calls.size(); i++) {
                ObjectNode call = toolCalls.addObject();
                call.put("id", "call_" + i);
                call.put("type", "function");
                call.putObject("function").put("name", calls.get(i)[0]).put("arguments", calls.get(i)[1]);
            }
        }
        if (message != null) {
            message.forEach((key, value) -> m.set(key, Json.node(value)));
        }
        if (usage != null) {
            body.set("usage", usage);
        }
        return body;
    }

    static ObjectNode chat() {
        return chat(List.of(), null, null, null, null);
    }

    static ObjectNode chatCall(String name, String arguments) {
        List<String[]> calls = new ArrayList<>();
        calls.add(new String[]{name, arguments});
        return chat(calls, null, null, null, null);
    }

    /** A /v1/systemone response body. */
    static ObjectNode answer(Double function, Double arguments, String model) {
        ObjectNode body = Json.object();
        body.put("model", model);
        ObjectNode answers = body.putObject("answers");
        answers.putObject("function_correct").put("type", "noul").put("noul", function);
        if (arguments != null) {
            answers.putObject("arguments_correct").put("type", "noul").put("noul", arguments);
        }
        body.putObject("usage").put("input_tokens", 300).put("output_tokens", 0);
        return body;
    }

    static ObjectNode answer(double function, Double arguments) {
        return answer(function, arguments, "jev-1.13.0");
    }

    /** Jev over a mock: arguments are right only for {@code goodCity}; a no call scores {@code noCall}. */
    static final class ScriptedJev {
        final List<JsonNode> requests = new ArrayList<>();
        final Jev jev;

        ScriptedJev() {
            this("Seoul", 0.95, 0.05);
        }

        ScriptedJev(String goodCity, double function, double noCall) {
            Mock mock = new Mock(request -> {
                JsonNode body = request.json();
                requests.add(body);
                JsonNode calls = body.get("state").get("proposed_calls");
                if (calls.isEmpty()) {
                    return HttpTransport.Response.json(200, answer(noCall, null));
                }
                JsonNode city = calls.get(0).get("arguments").get("city");
                double arguments = city != null && goodCity.equals(city.asText()) ? 0.95 : 0.05;
                return HttpTransport.Response.json(200, answer(function, arguments));
            });
            this.jev = Jev.builder().apiKey("jev-key").transport(mock).sleeper(NO_SLEEP).build();
        }
    }

    /** A generator stand-in that returns scripted candidates and records the history it saw. */
    static final class ScriptedModel implements Generator {
        private final Deque<List<Call>> proposals = new ArrayDeque<>();
        final List<ArrayNode> histories = new ArrayList<>();
        boolean closed;

        @SafeVarargs
        ScriptedModel(List<Call>... proposals) {
            if (proposals == null) {
                this.proposals.add(UNREADABLE);
                return;
            }
            for (List<Call> proposal : proposals) {
                this.proposals.add(proposal == null ? UNREADABLE : proposal);
            }
        }

        static final List<Call> UNREADABLE = new ArrayList<>();

        @Override
        public Proposal generate(ArrayNode messages, ArrayNode tools, ArrayNode history) {
            histories.add(history.deepCopy());
            List<Call> calls = proposals.removeFirst();
            Map<String, Long> usage = new LinkedHashMap<>();
            usage.put("input_tokens", 10L);
            usage.put("output_tokens", 2L);
            if (calls == UNREADABLE) {
                return Proposal.builder().calls(null).incomplete("length").parseErrors(List.of("cut off"))
                        .usage(usage).build();
            }
            return Proposal.builder().calls(calls).content(calls.isEmpty() ? "No tool fits." : null).usage(usage)
                    .build();
        }

        @Override
        public void close() {
            closed = true;
        }
    }
}
