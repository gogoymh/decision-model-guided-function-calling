package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.TextNode;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The Chat Completions protocol shared by {@link OpenAICompatible}, {@link OpenAI}, {@link XAI},
 * {@link OpenRouter}, {@link Meta} and {@link AzureOpenAI}.
 */
public abstract class ChatCompletionsProvider extends Provider {
    protected final String reasoningEffort;
    protected final Boolean parallelToolCalls;
    protected final String maxTokensField;
    protected final String path;

    protected ChatCompletionsProvider(ChatBuilder<?, ?> b, Kind kind, String defaultMaxTokensField, String baseUrl) {
        super(b, kind, b.model, baseUrl);
        if (b.path == null || !b.path.startsWith("/")) {
            throw new IllegalArgumentException("path must start with '/'.");
        }
        this.reasoningEffort = b.reasoningEffort;
        this.parallelToolCalls = b.parallelToolCalls;
        this.maxTokensField = b.maxTokensField != null && !b.maxTokensField.isEmpty() ? b.maxTokensField
                : defaultMaxTokensField;
        this.path = b.path;
    }

    public String reasoningEffort() {
        return reasoningEffort;
    }

    @Override
    public String url() {
        return baseUrl + path;
    }

    @Override
    protected ObjectNode payload(ArrayNode messages, ArrayNode tools, ArrayNode history) {
        ObjectNode payload = Json.object();
        payload.put("model", model);
        ArrayNode sent = payload.putArray("messages");
        ObjectNode system = sent.addObject();
        system.put("role", "system");
        system.put("content", systemPrompt);
        sent.addAll(Prompts.withHistory(messages, history));
        ArrayNode functions = Prompts.functionsOf(tools);
        if (!functions.isEmpty()) {  // a request without functions carries no tool settings
            ArrayNode declared = payload.putArray("tools");
            for (JsonNode function : functions) {
                ObjectNode tool = declared.addObject();
                tool.put("type", "function");
                tool.set("function", function);
            }
            payload.put("tool_choice", "auto");
            if (parallelToolCalls != null) {
                payload.put("parallel_tool_calls", parallelToolCalls);
            }
        }
        if (maxOutputTokens != null && maxOutputTokens != 0) {
            payload.put(maxTokensField, maxOutputTokens);
        }
        if (reasoningEffort != null && !reasoningEffort.isEmpty()) {
            payload.put("reasoning_effort", reasoningEffort);
        }
        if (temperature != null) {
            payload.set("temperature", Json.node(temperature));
        }
        return payload;
    }

    @Override
    protected Decoded decode(ObjectNode data) {
        if (!Py.isNone(data.get("error"))) {
            throw new ProviderException("The response contained a provider error.");
        }
        JsonNode choices = data.get("choices");
        if (choices == null || !choices.isArray() || choices.isEmpty() || !choices.get(0).isObject()) {
            throw new ProviderException("The response did not contain a choices list.");
        }
        JsonNode choice = choices.get(0);
        String finish = Py.member(choice.get("finish_reason"));
        if ("length".equals(finish) || "content_filter".equals(finish)) {
            return Decoded.incomplete(finish);
        }
        JsonNode message = choice.get("message");
        boolean assistant = message != null && message.isObject()
                && (!message.has("role") || "assistant".equals(Py.text(message.get("role"))));
        if (!assistant) {
            throw new ProviderException("The response did not contain an assistant message.");
        }
        if (Py.truthy(message.get("refusal"))) {
            return Decoded.incomplete("refusal");
        }
        List<Call> calls = new ArrayList<>();
        List<String> errors = new ArrayList<>();
        List<String> text = new ArrayList<>();
        JsonNode nativeCalls = Py.truthy(message.get("tool_calls")) ? message.get("tool_calls") : Json.array();
        if (!nativeCalls.isArray()) {
            throw new ProviderException("The response tool_calls must be a list.");
        }
        for (JsonNode item : nativeCalls) {
            if (!item.isObject() || item.get("function") == null || !item.get("function").isObject()) {
                throw new ProviderException("The response contained an invalid function-call object.");
            }
            JsonNode function = item.get("function");
            JsonNode arguments = function.has("arguments") ? function.get("arguments")
                    : TextNode.valueOf("{}");
            appendCall(calls, errors, function.get("name"), arguments, true);
        }
        JsonNode content = message.get("content");
        if (content != null && content.isTextual() && !content.textValue().isEmpty()) {
            text.add(content.textValue());
        } else if (!Py.isNone(content) && !content.isTextual()) {
            throw new ProviderException("The message content was not text.");
        }
        return Decoded.complete(calls, errors, text);
    }

    @Override
    protected Map<String, Long> usage(ObjectNode data) {
        JsonNode raw = data.get("usage");
        if (raw == null || !raw.isObject()) {
            return new LinkedHashMap<>();
        }
        JsonNode details = raw.get("prompt_tokens_details");
        Map<String, Long> counts = new LinkedHashMap<>();
        counts.put("input_tokens", count(raw.get("prompt_tokens")));
        counts.put("cached_input_tokens", details != null && details.isObject() ? count(details.get("cached_tokens"))
                : null);
        counts.put("output_tokens", count(raw.get("completion_tokens")));
        return withoutNulls(counts);
    }

    @Override
    protected boolean filteredRequest(int statusCode, String code) {
        return statusCode == 400 && ("content_filter".equals(code) || "invalid_prompt".equals(code));
    }

    /**
     * Options of every Chat Completions provider.
     *
     * @param <B> the builder type
     * @param <P> the provider it builds
     */
    public abstract static class ChatBuilder<B extends ChatBuilder<B, P>, P extends ChatCompletionsProvider>
            extends Provider.Builder<B, P> {
        protected String reasoningEffort;
        protected Boolean parallelToolCalls = false;
        protected String maxTokensField;
        protected String path = "/chat/completions";

        protected ChatBuilder(String model) {
            super(model);
        }

        /** {@code reasoning_effort} (for example {@code "none"} for gpt-5-nano in the paper). */
        public B reasoningEffort(String reasoningEffort) {
            this.reasoningEffort = reasoningEffort;
            return self();
        }

        /** {@code parallel_tool_calls}; {@code false} by default, {@code null} to omit it. */
        public B parallelToolCalls(Boolean parallelToolCalls) {
            this.parallelToolCalls = parallelToolCalls;
            return self();
        }

        /** The body field of the output limit ({@code max_completion_tokens} or {@code max_tokens}). */
        public B maxTokensField(String maxTokensField) {
            this.maxTokensField = maxTokensField;
            return self();
        }

        /** The path after the base URL ({@code /chat/completions}). */
        public B path(String path) {
            this.path = path;
            return self();
        }
    }
}
