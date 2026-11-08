package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.TextNode;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Anthropic Messages API. Key: {@code ANTHROPIC_API_KEY} (base URL: {@code ANTHROPIC_BASE_URL}, optional). */
public final class Anthropic extends Provider {
    static final Kind KIND = new Kind("anthropic", "https://api.anthropic.com/v1", List.of("ANTHROPIC_BASE_URL"),
            List.of("ANTHROPIC_API_KEY"), true);
    /** The {@code anthropic-version} header. */
    public static final String VERSION = "2023-06-01";

    private Anthropic(Builder b) {
        super(b, KIND, b.model, null);
    }

    /** A provider for {@code model} with the key from the environment. */
    public Anthropic(String model) {
        this(builder(model));
    }

    public static Builder builder(String model) {
        return new Builder(model);
    }

    @Override
    public String url() {
        return baseUrl + "/messages";
    }

    @Override
    protected Map<String, String> authHeaders() {
        Map<String, String> headers = new LinkedHashMap<>();
        headers.put("x-api-key", apiKey());
        headers.put("anthropic-version", VERSION);
        return headers;
    }

    @Override
    protected ObjectNode payload(ArrayNode messages, ArrayNode tools, ArrayNode history) {
        List<String> system = new ArrayList<>();
        system.add(systemPrompt);
        ArrayNode conversation = Json.array();
        for (JsonNode message : Prompts.withHistory(messages, history)) {
            String role = Py.text(message.get("role"));
            if ("system".equals(role) || "developer".equals(role)) {
                if (!message.get("content").isTextual()) {
                    throw new IllegalArgumentException("Anthropic needs text system messages.");
                }
                system.add(message.get("content").textValue());
            } else if ("user".equals(role) || "assistant".equals(role)) {
                conversation.add(message);
            } else {
                throw new IllegalArgumentException("Anthropic accepts only user and assistant conversation messages.");
            }
        }
        ObjectNode payload = Json.object();
        payload.put("model", model);
        payload.put("system", String.join("\n\n", system));
        payload.set("messages", conversation);
        payload.put("max_tokens", maxOutputTokens != null && maxOutputTokens != 0 ? maxOutputTokens : 4096);
        ArrayNode functions = Prompts.functionsOf(tools);
        if (!functions.isEmpty()) {
            ArrayNode declared = payload.putArray("tools");
            for (JsonNode function : functions) {
                ObjectNode tool = declared.addObject();
                tool.set("name", function.get("name").deepCopy());
                tool.set("description", function.has("description") ? function.get("description").deepCopy()
                        : TextNode.valueOf(""));
                tool.set("input_schema", function.has("parameters") ? function.get("parameters").deepCopy()
                        : emptyParameters());
            }
            ObjectNode choice = payload.putObject("tool_choice");
            choice.put("type", "auto");
            choice.put("disable_parallel_tool_use", true);
        }
        if (temperature != null) {
            payload.set("temperature", Json.node(temperature));
        }
        return payload;
    }

    static ObjectNode emptyParameters() {
        ObjectNode parameters = Json.object();
        parameters.put("type", "object");
        parameters.putObject("properties");
        return parameters;
    }

    @Override
    protected Decoded decode(ObjectNode data) {
        if ("error".equals(Py.text(data.get("type"))) || !Py.isNone(data.get("error"))) {
            throw new ProviderException("The response contained a provider error.");
        }
        if (!"assistant".equals(Py.text(data.get("role"))) || data.get("content") == null
                || !data.get("content").isArray()) {
            throw new ProviderException("The Anthropic response was not an assistant message.");
        }
        JsonNode details = data.get("stop_details");
        String reason = Py.member(data.get("stop_reason"));
        if ("max_tokens".equals(reason) || "model_context_window_exceeded".equals(reason)) {
            return Decoded.incomplete("length");
        }
        if ("refusal".equals(reason) || (details != null && details.isObject()
                && "refusal".equals(Py.text(details.get("type"))))) {
            return Decoded.incomplete("refusal");
        }
        List<Call> calls = new ArrayList<>();
        List<String> errors = new ArrayList<>();
        List<String> text = new ArrayList<>();
        for (JsonNode block : data.get("content")) {
            if (!block.isObject()) {
                throw new ProviderException("An Anthropic content block was malformed.");
            }
            String type = Py.text(block.get("type"));
            if ("tool_use".equals(type)) {
                appendCall(calls, errors, block.get("name"), block.get("input"), false);
            } else if ("text".equals(type) && block.get("text") != null && block.get("text").isTextual()
                    && !block.get("text").textValue().isEmpty()) {
                text.add(block.get("text").textValue());
            }
        }
        return Decoded.complete(calls, errors, text);
    }

    @Override
    protected Map<String, Long> usage(ObjectNode data) {
        JsonNode raw = data.get("usage");
        if (raw == null || !raw.isObject()) {
            return new LinkedHashMap<>();
        }
        Long read = count(raw.get("cache_read_input_tokens"));
        long cached = read == null ? 0 : read;
        Long written = count(raw.get("cache_creation_input_tokens"));
        long creation = written == null ? 0 : written;
        Long base = count(raw.get("input_tokens"));
        Map<String, Long> result = new LinkedHashMap<>();
        result.put("cached_input_tokens", cached);
        result.put("output_tokens", count(raw.get("output_tokens")));
        if (base != null) {
            result.put("input_tokens", base + cached + creation);
        }
        return withoutNulls(result);
    }

    /** Builds an {@link Anthropic}. */
    public static final class Builder extends Provider.Builder<Builder, Anthropic> {
        Builder(String model) {
            super(model);
        }

        @Override
        protected Builder self() {
            return this;
        }

        @Override
        public Anthropic build() {
            return new Anthropic(this);
        }
    }
}
