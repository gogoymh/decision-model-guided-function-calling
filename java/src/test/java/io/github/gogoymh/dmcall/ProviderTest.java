package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static io.github.gogoymh.dmcall.Support.Mock;
import static io.github.gogoymh.dmcall.Support.SECRET;
import static io.github.gogoymh.dmcall.Support.chat;
import static io.github.gogoymh.dmcall.Support.chatCall;
import static io.github.gogoymh.dmcall.Support.exampleMessages;
import static io.github.gogoymh.dmcall.Support.exampleTools;
import static io.github.gogoymh.dmcall.Support.history;
import static io.github.gogoymh.dmcall.Support.seoul;
import static io.github.gogoymh.dmcall.Support.weather;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Generators over plain HTTP: requests, decoding, unreadable outputs, keys and errors (test_providers.py). */
class ProviderTest {
    @BeforeEach
    void isolate() {
        Support.isolate();
    }

    private static ObjectNode usage() {
        return (ObjectNode) Json.parse("{\"prompt_tokens\": 120, \"completion_tokens\": 9, \"prompt_tokens_details\":"
                + " {\"cached_tokens\": 64}}");
    }

    @Test
    void openaiRequestAndDecoding() {
        ObjectNode body = chatCall("get_weather", "{\"city\": \"Seoul\"}");
        body.set("usage", usage());
        Mock mock = Mock.serving(body);
        OpenAI model = OpenAI.builder("gpt-5-nano").apiKey(SECRET).reasoningEffort("none").transport(mock).build();
        Proposal proposal = model.generate(exampleMessages(), exampleTools(), history());
        HttpTransport.Request request = mock.requests.get(0);
        JsonNode sent = request.json();
        assertEquals("https://api.openai.com/v1/chat/completions", request.url());
        assertEquals("Bearer " + SECRET, request.header("authorization"));
        assertEquals(Json.parse("{\"role\": \"system\", \"content\": " + Json.dumps(Json.node(Prompts.SYSTEM_PROMPT))
                + "}"), sent.get("messages").get(0));
        assertTrue(sent.get("messages").get(1).get("content").textValue()
                .startsWith("Accumulated revision history (oldest first; proposals NOT executed)"));
        // The original request follows, unchanged.
        assertEquals(exampleMessages(), Json.node(List.of(sent.get("messages").get(2), sent.get("messages").get(3))));
        assertEquals(exampleTools(), sent.get("tools"));
        assertEquals("auto", sent.get("tool_choice").textValue());
        assertFalse(sent.get("parallel_tool_calls").booleanValue());
        assertEquals(4096, sent.get("max_completion_tokens").intValue());
        assertEquals("none", sent.get("reasoning_effort").textValue());
        assertEquals(seoul(), proposal.calls());
        assertNull(proposal.incomplete());
        assertEquals(Map.of("input_tokens", 120L, "cached_input_tokens", 64L, "output_tokens", 9L), proposal.usage());
        assertEquals("resolved-model", proposal.model());
        assertEquals("openai", proposal.provider());
        assertFalse(model.toString().contains(SECRET));
    }

    @Test
    void historyListsRoundsOldestFirst() {
        Mock mock = Mock.serving(chat());
        OpenAI.builder("m").apiKey("k").transport(mock).build().generate(exampleMessages(), exampleTools(), history());
        String text = mock.body(0).get("messages").get(1).get("content").textValue();
        String records = text.substring(text.indexOf('\n') + 1, text.lastIndexOf("\nUse these previous judgments"));
        JsonNode parsed = Json.parse(records);
        assertEquals(1, parsed.get(0).get("round").intValue());
        assertEquals(2, parsed.get(1).get("round").intValue());
        assertEquals(history().get(0).get("calls"), parsed.get(0).get("calls"));
        assertEquals(Json.array(), parsed.get(1).get("calls"));
        assertTrue(text.endsWith("when answering the original request below."));
    }

    @Test
    void openaiOutcomes() {
        Object[][] cases = {
            {chat(List.of(), "No function applies.", null, null, null), List.of(), null},
            {chat(), List.of(), null},
            {chat(List.of(), null, "length", null, null), null, "length"},
            {chat(List.of(), null, "content_filter", null, null), null, "content_filter"},
            {chat(List.of(), null, null, null, Map.of("refusal", "I can't help with that.")), null, "refusal"},
            {chatCall("get_weather", "{not json"), List.of(new Call("get_weather", null)), null},
        };
        for (Object[] c : cases) {
            Proposal proposal = OpenAI.builder("m").apiKey("k").transport(Mock.serving((JsonNode) c[0])).build()
                    .generate(exampleMessages(), exampleTools(), null);
            assertEquals(c[1], proposal.calls());
            assertEquals(c[2], proposal.incomplete());
            if (c[2] != null || (proposal.calls() != null && !proposal.calls().isEmpty())) {
                assertFalse(proposal.parseErrors().isEmpty());
            }
        }
    }

    @Test
    void filteredPromptIsAnUnreadableCandidateNotAnError() {
        Mock mock = Mock.always(HttpTransport.Response.json(400,
                Json.parse("{\"error\": {\"code\": \"content_filter\", \"message\": \"filtered\"}}")));
        Proposal proposal = OpenAI.builder("m").apiKey("k").transport(mock).build()
                .generate(exampleMessages(), exampleTools(), null);
        assertNull(proposal.calls());
        assertEquals("prompt_filter", proposal.incomplete());
        assertTrue(proposal.usage().isEmpty());
    }

    @Test
    void httpErrorsAreSanitizedAndRetriesBounded() {
        Mock failing = Mock.always(HttpTransport.Response.of(401, "{\"error\": {\"code\": \"invalid_api_key\","
                + " \"message\": \"" + SECRET + "\"}}", Map.of("x-request-id", "req-1")));
        ProviderException caught = assertThrows(ProviderException.class, () -> OpenAI.builder("m").apiKey(SECRET)
                .maxRetries(2).transport(failing).build().generate(exampleMessages(), exampleTools(), null));
        assertEquals(1, failing.requests.size());
        assertEquals(401, caught.statusCode());
        assertEquals("invalid_api_key", caught.providerCode());
        assertEquals("req-1", caught.requestId());
        assertFalse(caught.getMessage().contains(SECRET));
        List<Double> waits = new ArrayList<>();
        Mock retried = Mock.replies(HttpTransport.Response.of(429, null, Map.of("retry-after", "1")),
                HttpTransport.Response.of(503, null, null),
                HttpTransport.Response.json(200, chatCall("get_weather", "{\"city\": \"Seoul\"}")));
        Proposal proposal = OpenAI.builder("m").apiKey("k").maxRetries(2).transport(retried).sleeper(waits::add).build()
                .generate(exampleMessages(), exampleTools(), null);
        assertEquals(3, proposal.apiCalls());
        assertEquals(seoul(), proposal.calls());
        assertEquals(List.of(1.0, 2.0), waits);
        Mock exhausted = Mock.always(HttpTransport.Response.of(500, null, null));
        ProviderException error = assertThrows(ProviderException.class, () -> OpenAI.builder("m").apiKey("k")
                .maxRetries(1).transport(exhausted).sleeper(Support.NO_SLEEP).build()
                .generate(exampleMessages(), exampleTools(), null));
        assertTrue(error.getMessage().contains("HTTP 500"));
        Mock down = new Mock(request -> null);
        error = assertThrows(ProviderException.class, () -> OpenAI.builder("m").apiKey("k").maxRetries(1)
                .transport(down).sleeper(Support.NO_SLEEP).build().generate(exampleMessages(), exampleTools(), null));
        assertTrue(error.getMessage().contains("transport failed after 2 attempts"));
    }

    @Test
    void protocolMismatchesAreSanitized() {
        for (String body : new String[]{"not json", "{\"choices\": [{\"finish_reason\": [1]}]}"}) {
            ProviderException error = assertThrows(ProviderException.class, () -> OpenAI.builder("m").apiKey("k")
                    .transport(Mock.always(HttpTransport.Response.of(200, body, null))).build()
                    .generate(exampleMessages(), exampleTools(), null));
            assertEquals("The openai response did not match its protocol.", error.getMessage());
            assertEquals(200, error.statusCode());
        }
    }

    @Test
    void keysComeFromOptionsOrEnvironment() {
        IllegalArgumentException missing = assertThrows(IllegalArgumentException.class, () -> new OpenAI("m"));
        assertTrue(missing.getMessage().contains("OPENAI_API_KEY"));
        Support.environment(Map.of("OPENAI_API_KEY", "from-env", "OPENAI_BASE_URL", "https://proxy.example.com/v1/"));
        Mock mock = Mock.serving(chat());
        OpenAI.builder("m").transport(mock).build().generate(exampleMessages(), exampleTools(), null);
        assertEquals("Bearer from-env", mock.requests.get(0).header("authorization"));
        assertEquals("https://proxy.example.com/v1/chat/completions", mock.requests.get(0).url());
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> OpenAI.builder("m").baseUrl("https://user:pass@example.com/v1").build());
        assertTrue(error.getMessage().contains("baseUrl"));
        error = assertThrows(IllegalArgumentException.class, () -> OpenAI.builder("m").apiKey("a\nb").build());
        assertTrue(error.getMessage().contains("single-line"));
    }

    @Test
    void metaMuseRequest() {
        Support.environment(Map.of("MODEL_API_KEY", SECRET));
        Mock mock = Mock.serving(chatCall("get_weather", "{\"city\": \"Seoul\"}"));
        Meta model = Meta.builder("muse-spark-1.3").transport(mock).build();  // the key comes from MODEL_API_KEY
        Proposal proposal = model.generate(exampleMessages(), exampleTools(), history());
        JsonNode body = mock.body(0);
        assertEquals("https://api.meta.ai/v1/chat/completions", mock.requests.get(0).url());
        assertEquals("Bearer " + SECRET, mock.requests.get(0).header("authorization"));
        assertEquals("muse-spark-1.3", body.get("model").textValue());
        assertEquals("auto", body.get("tool_choice").textValue());
        assertEquals(4096, body.get("max_completion_tokens").intValue());
        // Meta rejects reasoning_effort "none" for Muse models; neither option is sent unless set.
        assertFalse(body.has("parallel_tool_calls"));
        assertFalse(body.has("reasoning_effort"));
        assertEquals(seoul(), proposal.calls());
        assertEquals("meta", proposal.provider());
        assertFalse(model.toString().contains(SECRET));
        Support.isolate();
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> new Meta("muse-spark-1.3"));
        assertTrue(error.getMessage().contains("MODEL_API_KEY"));
        Provider built = ((Meta.Builder) Providers.builder("meta:muse-spark-1.3")).apiKey("k").reasoningEffort("low")
                .parallelToolCalls(true).build();
        assertTrue(built instanceof Meta);
        ObjectNode payload = built.payload(Json.array(), exampleTools(), null);
        assertEquals("low", payload.get("reasoning_effort").textValue());
        assertTrue(payload.get("parallel_tool_calls").booleanValue());
    }

    @Test
    void openaiCompatibleHosts() {
        Support.environment(Map.of("XAI_API_KEY", "k", "OPENROUTER_API_KEY", "k"));
        Mock xai = Mock.serving(chat());
        XAI.builder("m").transport(xai).build().generate(exampleMessages(), exampleTools(), null);
        assertEquals("https://api.x.ai/v1/chat/completions", xai.requests.get(0).url());
        assertEquals(4096, xai.body(0).get("max_tokens").intValue());
        Mock router = Mock.serving(chat());
        OpenRouter.builder("m").transport(router).build().generate(exampleMessages(), exampleTools(), null);
        assertEquals("https://openrouter.ai/api/v1/chat/completions", router.requests.get(0).url());
        assertEquals(4096, router.body(0).get("max_tokens").intValue());
    }

    @Test
    void anyOpenAICompatibleServerNeedsNoKey() {
        Mock mock = Mock.serving(chat());
        OpenAICompatible.builder("gemma-4-31b-it").baseUrl("http://localhost:8000/v1").transport(mock).build()
                .generate((ArrayNode) Json.parse("[{\"role\": \"user\", \"content\": \"hi\"}]"), Json.array(), null);
        JsonNode body = mock.body(0);
        assertEquals("http://localhost:8000/v1/chat/completions", mock.requests.get(0).url());
        assertNull(mock.requests.get(0).header("authorization"));
        for (String key : new String[]{"tools", "tool_choice", "parallel_tool_calls"}) {
            assertFalse(body.has(key), key);  // no functions, no tool settings
        }
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> OpenAICompatible.builder("m").build());
        assertTrue(error.getMessage().contains("baseUrl"));
    }

    @Test
    void azureDeploymentUrlAndHeader() {
        Support.environment(Map.of("AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com/", "AZURE_OPENAI_API_KEY",
                SECRET));
        Mock mock = Mock.serving(chat());
        AzureOpenAI.builder("my deployment").transport(mock).build().generate(exampleMessages(), exampleTools(), null);
        assertEquals("https://res.openai.azure.com/openai/deployments/my%20deployment/chat/completions"
                + "?api-version=2024-12-01-preview", mock.requests.get(0).url());
        assertEquals(SECRET, mock.requests.get(0).header("api-key"));
        assertNull(mock.requests.get(0).header("authorization"));
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> AzureOpenAI.builder("d").apiVersion("latest").build());
        assertTrue(error.getMessage().contains("apiVersion"));
        Support.environment(Map.of("AZURE_OPENAI_API_KEY", SECRET));
        error = assertThrows(IllegalArgumentException.class, () -> new AzureOpenAI("d"));
        assertTrue(error.getMessage().contains("AZURE_OPENAI_ENDPOINT"));
    }

    @Test
    void anthropicRequestAndDecoding() {
        ObjectNode body = (ObjectNode) Json.parse("{\"type\": \"message\", \"role\": \"assistant\", \"model\":"
                + " \"claude-x\", \"stop_reason\": \"tool_use\", \"content\": [{\"type\": \"thinking\", \"thinking\":"
                + " \"...\"}, {\"type\": \"text\", \"text\": \"Checking.\"}, {\"type\": \"tool_use\", \"id\": \"t1\","
                + " \"name\": \"get_weather\", \"input\": {\"city\": \"Seoul\"}}], \"usage\": {\"input_tokens\": 50,"
                + " \"cache_read_input_tokens\": 30, \"cache_creation_input_tokens\": 20, \"output_tokens\": 7}}");
        Mock mock = Mock.serving(body);
        Proposal proposal = Anthropic.builder("claude-x").apiKey(SECRET).transport(mock).build()
                .generate(exampleMessages(), exampleTools(), history());
        JsonNode request = mock.body(0);
        assertEquals("https://api.anthropic.com/v1/messages", mock.requests.get(0).url());
        assertEquals(SECRET, mock.requests.get(0).header("x-api-key"));
        assertEquals("2023-06-01", mock.requests.get(0).header("anthropic-version"));
        assertEquals(Prompts.SYSTEM_PROMPT + "\n\nBe brief.", request.get("system").textValue());
        assertEquals("user", request.get("messages").get(0).get("role").textValue());
        assertEquals("user", request.get("messages").get(1).get("role").textValue());
        assertEquals(weather().get("function").get("parameters"), request.get("tools").get(0).get("input_schema"));
        assertEquals(Json.parse("{\"type\": \"auto\", \"disable_parallel_tool_use\": true}"), request.get("tool_choice"));
        assertEquals(seoul(), proposal.calls());
        assertEquals("Checking.", proposal.content());
        assertEquals(Map.of("input_tokens", 100L, "cached_input_tokens", 30L, "output_tokens", 7L), proposal.usage());
        for (String[] c : new String[][]{{"max_tokens", "length"}, {"refusal", "refusal"}}) {
            ObjectNode cut = body.deepCopy();
            cut.put("stop_reason", c[0]);
            cut.putArray("content");
            assertEquals(c[1], Anthropic.builder("c").apiKey("k").transport(Mock.serving(cut)).build()
                    .generate(exampleMessages(), exampleTools(), null).incomplete());
        }
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class, () -> Anthropic.builder("c")
                .apiKey("k").transport(Mock.serving(body)).build()
                .generate((ArrayNode) Json.parse("[{\"role\": \"tool\", \"content\": \"x\"}]"), exampleTools(), null));
        assertTrue(error.getMessage().contains("only user and assistant"));
    }

    @Test
    void geminiRequestAndDecoding() {
        JsonNode body = Json.parse("{\"modelVersion\": \"gemini-x-001\", \"candidates\": [{\"finishReason\": \"STOP\","
                + " \"content\": {\"role\": \"model\", \"parts\": [{\"text\": \"thinking\", \"thought\": true},"
                + " {\"functionCall\": {\"name\": \"get_weather\", \"args\": {\"city\": \"Seoul\"}}}]}}],"
                + " \"usageMetadata\": {\"promptTokenCount\": 80, \"candidatesTokenCount\": 6, \"thoughtsTokenCount\":"
                + " 40}}");
        Mock mock = Mock.serving(body);
        ArrayNode messages = exampleMessages();
        messages.addObject().put("role", "assistant").put("content", "Which unit?");
        messages.addObject().put("role", "user").put("content", "Celsius.");
        Proposal proposal = Gemini.builder("models/gemini-x").apiKey(SECRET).temperature(0).thinkingBudget(0)
                .transport(mock).build().generate(messages, exampleTools(), null);
        JsonNode request = mock.body(0);
        assertEquals("https://generativelanguage.googleapis.com/v1beta/models/gemini-x:generateContent",
                mock.requests.get(0).url());
        assertEquals(SECRET, mock.requests.get(0).header("x-goog-api-key"));
        assertEquals(Prompts.SYSTEM_PROMPT + "\n\nBe brief.",
                request.get("systemInstruction").get("parts").get(0).get("text").textValue());
        List<String> roles = new ArrayList<>();
        request.get("contents").forEach(content -> roles.add(content.get("role").textValue()));
        assertEquals(List.of("user", "model", "user"), roles);
        JsonNode declaration = request.get("tools").get(0).get("functionDeclarations").get(0);
        assertEquals(weather().get("function").get("parameters"), declaration.get("parametersJsonSchema"));
        assertFalse(declaration.has("parameters"));
        assertEquals(Json.parse("{\"functionCallingConfig\": {\"mode\": \"AUTO\"}}"), request.get("toolConfig"));
        assertEquals(Json.parse("{\"maxOutputTokens\": 4096, \"temperature\": 0, \"thinkingConfig\":"
                + " {\"thinkingBudget\": 0}}"), request.get("generationConfig"));
        assertEquals(seoul(), proposal.calls());
        assertNull(proposal.content());
        assertEquals("gemini-x-001", proposal.model());
        assertEquals(Map.of("input_tokens", 80L, "output_tokens", 46L), proposal.usage());
    }

    @Test
    void geminiUnreadableOutputs() {
        String[][] cases = {
            {"{\"candidates\": [{\"finishReason\": \"MAX_TOKENS\", \"content\": {\"parts\": []}}]}", "length"},
            {"{\"candidates\": [{\"finishReason\": \"SAFETY\"}]}", "content_filter"},
            {"{\"candidates\": [{\"finishReason\": \"MALFORMED_FUNCTION_CALL\"}]}", "malformed_call"},
            {"{\"promptFeedback\": {\"blockReason\": \"SAFETY\"}}", "prompt_filter"},
        };
        for (String[] c : cases) {
            Proposal proposal = Gemini.builder("g").apiKey("k").transport(Mock.serving(Json.parse(c[0]))).build()
                    .generate(exampleMessages(), exampleTools(), null);
            assertNull(proposal.calls());
            assertEquals(c[1], proposal.incomplete());
        }
    }

    @Test
    void providerSpecs() {
        Support.environment(Map.of("OPENAI_API_KEY", "k", "ANTHROPIC_API_KEY", "k", "GEMINI_API_KEY", "k",
                "XAI_API_KEY", "k", "OPENROUTER_API_KEY", "k", "MODEL_API_KEY", "k"));
        Map<String, Class<?>> expected = Map.of("openai:gpt-5-nano", OpenAI.class, "anthropic:claude-x",
                Anthropic.class, "claude:c", Anthropic.class, "gemini:g", Gemini.class, "google:g", Gemini.class,
                "xai:grok-4", XAI.class, "grok:grok-4", XAI.class, "openrouter:meta/llama", OpenRouter.class,
                "meta:muse-spark-1.3", Meta.class);
        expected.forEach((spec, cls) -> {
            Provider built = Providers.fromSpec(spec);
            assertSame(cls, built.getClass(), spec);
            assertEquals(spec.substring(spec.indexOf(':') + 1), built.model());
            built.close();
        });
        Provider azure = ((AzureOpenAI.Builder) Providers.builder("azure:dep")).endpoint("https://r.openai.azure.com")
                .apiKey("k").build();
        assertSame(AzureOpenAI.class, azure.getClass());
        assertEquals("dep", azure.model());
        for (String bad : new String[]{"gpt-5-nano", "openai:", "mystery:model"}) {
            assertThrows(IllegalArgumentException.class, () -> Providers.fromSpec(bad), bad);
        }
    }

    @Test
    void azureSpecNeedsAnEndpoint() {
        Support.environment(Map.of("AZURE_OPENAI_ENDPOINT", "https://r.openai.azure.com"));
        Provider built = Providers.builder("azure:dep").apiKey("k").build();
        assertTrue(built instanceof AzureOpenAI);
    }
}
