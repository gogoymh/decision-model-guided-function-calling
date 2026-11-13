package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.math.BigInteger;
import java.util.List;
import java.util.Map;

import static io.github.gogoymh.dmcall.Support.json;
import static io.github.gogoymh.dmcall.Support.seoul;
import static io.github.gogoymh.dmcall.Support.weather;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Tool definitions in every format, calls from any SDK (test_tools.py), and Python-compatible JSON text. */
class ToolsAndJsonTest {
    @BeforeEach
    void isolate() {
        Support.isolate();
    }

    @Test
    void definitionFormats() {
        ObjectNode function = (ObjectNode) weather().get("function");
        JsonNode parameters = function.get("parameters");
        ObjectNode flatWithType = function.deepCopy();
        ObjectNode typed = Json.object();
        typed.put("type", "function");
        typed.setAll(flatWithType);
        ObjectNode anthropic = Json.object();
        anthropic.put("name", "get_weather");
        anthropic.put("description", "Current weather for a city.");
        anthropic.set("input_schema", parameters);
        JsonNode gemini = json("{\"name\": \"get_weather\", \"description\": \"Current weather for a city.\","
                + " \"parameters\": {\"type\": \"OBJECT\", \"properties\": {\"city\": {\"type\": \"STRING\"}, \"unit\":"
                + " {\"type\": \"STRING\", \"enum\": [\"celsius\", \"fahrenheit\"]}}, \"required\": [\"city\"]}}");
        for (JsonNode definition : List.of(weather(), function, typed, anthropic, gemini)) {
            assertEquals(weather(), Tool.of(definition).definition(), definition.toString());
        }
        Tool tool = Tool.function("get_weather", "Current weather for a city.", parameters);
        assertEquals(weather(), tool.definition());
        assertEquals("get_weather", tool.name());
        Tool noParameters = Tool.function("ping", "Ping.", null);
        assertEquals(json("{\"type\": \"object\", \"properties\": {}}"), noParameters.parameters());
    }

    @Test
    void badTools() {
        Object[] bad = {"get_weather", Map.of("name", "", "parameters", Map.of("type", "object")),
                Map.of("name", "x", "parameters", Map.of("type", "array")),
                Map.of("name", "x", "parameters", Map.of("type", "object", "properties",
                        Map.of("a", Map.of("type", "nonsense")))),
                Map.of("name", "x", "description", 5),
                Map.of("functionDeclarations", List.of())};
        for (Object definition : bad) {
            assertThrows(IllegalArgumentException.class, () -> Tool.of(definition), String.valueOf(definition));
        }
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> Tools.toolset(List.of(weather(), weather()), null));
        assertTrue(error.getMessage().contains("Two tools"));
        error = assertThrows(IllegalArgumentException.class,
                () -> Tools.toolset(List.of(weather()), Map.of("nope", arguments -> null)));
        assertTrue(error.getMessage().contains("not one of the tools"));
        error = assertThrows(IllegalArgumentException.class,
                () -> Tool.of(Map.of("name", "x", "parameters", Map.of("type", "object", "properties",
                        Map.of("a", Map.of("type", "nonsense"))))));
        assertEquals("Tool 'x': parameters are not a valid JSON Schema (SchemaError).", error.getMessage());
    }

    @Test
    void toolsetPairsHandlers() {
        Tool named = Tool.function("weather", "Weather now.", Map.of("type", "object", "properties",
                Map.of("city", Map.of("type", "string"))), arguments -> "ok");
        Tools.Toolset toolset = Tools.toolset(List.of(named, weather()), Map.of("get_weather", arguments -> "fn"));
        assertEquals("weather", toolset.schemas().get(0).get("function").get("name").textValue());
        assertEquals("get_weather", toolset.schemas().get(1).get("function").get("name").textValue());
        assertEquals(java.util.Set.of("weather", "get_weather"), toolset.registry().keySet());
    }

    @Test
    void callsFromEverySdkShape() {
        String[] shapes = {
            "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Seoul\"}}",
            "{\"id\": \"c1\", \"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"arguments\":"
                    + " \"{\\\"city\\\": \\\"Seoul\\\"}\"}}",
            "{\"type\": \"function_call\", \"call_id\": \"c1\", \"name\": \"get_weather\", \"arguments\":"
                    + " \"{\\\"city\\\": \\\"Seoul\\\"}\"}",
            "{\"type\": \"tool_use\", \"id\": \"t1\", \"name\": \"get_weather\", \"input\": {\"city\": \"Seoul\"}}",
            "{\"functionCall\": {\"name\": \"get_weather\", \"args\": {\"city\": \"Seoul\"}}}",
        };
        for (String shape : shapes) {
            assertEquals(seoul(), Calls.normalize(List.of(json(shape))), shape);
        }
        assertEquals(List.of(), Calls.normalize(null));  // SDKs report no call as tool_calls = null
        assertEquals(List.of(new Call("f", null)), Calls.normalize(Map.of("name", "f", "arguments", "{oops")));
        assertEquals(seoul(), Calls.normalize(seoul()));
        assertThrows(IllegalArgumentException.class, () -> Calls.normalize(3));
    }

    @Test
    void callFormatting() {
        assertEquals("get_weather(city=\"서울\", days=2.0)",
                Call.of("get_weather", json("{\"city\": \"서울\", \"days\": 2.0}")).toString());
        assertEquals("get_weather(<malformed arguments>)", new Call("get_weather", null).toString());
        assertNull(new Call("x", json("null")).arguments());
    }

    // JSON text -------------------------------------------------------------------------------
    @Test
    void parsingKeepsNumberKinds() {
        assertTrue(json("1e16").isDouble());
        assertTrue(json("2.0").isDouble());
        assertTrue(json("2").isInt());
        assertEquals(new BigInteger("100000000000000000000"), json("100000000000000000000").bigIntegerValue());
        assertEquals("{\"b\": 1, \"a\": [1.0, 2]}", Json.dumps(json("{\"b\": 1, \"a\": [1.0, 2]}")));
        assertThrows(IllegalArgumentException.class, () -> Json.parse("{} trailing"));
        assertThrows(IllegalArgumentException.class, () -> Json.parse(""));
    }

    @Test
    void floatsUsePythonRepr() {
        Object[][] cases = {{1.0, "1.0"}, {1e16, "1e+16"}, {1e-5, "1e-05"}, {0.0001, "0.0001"}, {-0.0, "-0.0"},
                {1e15, "1000000000000000.0"}, {5e-324, "5e-324"}, {1.7976931348623157e308, "1.7976931348623157e+308"},
                {0.1 + 0.2, "0.30000000000000004"}, {123456789.125, "123456789.125"}, {1e22, "1e+22"},
                {Double.NaN, "nan"}, {Double.NEGATIVE_INFINITY, "-inf"}};
        for (Object[] c : cases) {
            assertEquals(c[1], Py.floatRepr((Double) c[0]));
        }
        assertEquals("[NaN, Infinity, -Infinity]",
                Json.dumps(Json.node(List.of(Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY))));
        assertFalse(Json.isFinite(Json.node(Map.of("x", List.of(Double.NaN)))));
    }

    @Test
    void ensureAsciiAndSortedKeysMatchPython() {
        JsonNode value = json("{\"b\": \"서울\\u007f\", \"a\": \"🌤\"}");
        assertEquals("{\"a\": \"\\ud83c\\udf24\", \"b\": \"\\uc11c\\uc6b8\\u007f\"}", Json.dumps(value, true, true));
        assertEquals("{\"b\": \"서울\u007f\", \"a\": \"🌤\"}", Json.dumps(value));
    }

    @Test
    void pythonReprOfValues() {
        assertEquals("{'a': [1, 2.5, None, True, \"it's\"]}",
                Py.repr(json("{\"a\": [1, 2.5, null, true, \"it's\"]}")));
        assertEquals("'tab\\there\\x00'", Py.strRepr("tab\there\u0000"));
    }

    @Test
    void valuesFromJavaObjects() {
        assertEquals("[1, 2.5, 0.1, true, null, \"x\"]", Json.dumps(Json.node(java.util.Arrays.asList(1, 2.5, 0.1f, true,
                null, "x"))));
        assertEquals("{\"role\": \"user\", \"content\": \"hi\"}", Json.dumps(Message.user("hi").toJson()));
    }

    @Test
    void secretsStayOutOfStrings() {
        HttpTransport.Request request = new HttpTransport.Request("POST", "https://x.example.com",
                Map.of("Authorization", "Bearer " + Support.SECRET), Support.SECRET.getBytes(), null);
        assertFalse(request.toString().contains(Support.SECRET));
        HttpTransport.Response response = HttpTransport.Response.of(500, Support.SECRET,
                Map.of("x-api-key", Support.SECRET));
        assertFalse(response.toString().contains(Support.SECRET));
        Provider provider = OpenAI.builder("m").apiKey(Support.SECRET).transport(Support.Mock.always(response))
                .sleeper(Support.NO_SLEEP).build();
        assertFalse(provider.toString().contains(Support.SECRET));
        ProviderException error = assertThrows(ProviderException.class, () -> provider.generate(
                Support.exampleMessages(), Support.exampleTools(), null));
        assertFalse(error.getMessage().contains(Support.SECRET));
    }
}
