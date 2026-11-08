package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.TextNode;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Google Gemini API ({@code generateContent} with function declarations). Key:
 * {@code GEMINI_API_KEY} or {@code GOOGLE_API_KEY}.
 */
public final class Gemini extends Provider {
    static final Kind KIND = new Kind("gemini", "https://generativelanguage.googleapis.com/v1beta", List.of(),
            List.of("GEMINI_API_KEY", "GOOGLE_API_KEY"), true);
    private static final Set<String> FILTERED = Set.of("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT",
            "SPII", "IMAGE_SAFETY", "LANGUAGE");

    private final Integer thinkingBudget;

    private Gemini(Builder b) {
        super(b, KIND, b.model != null && b.model.startsWith("models/") ? b.model.substring("models/".length())
                : b.model, null);
        this.thinkingBudget = b.thinkingBudget;
    }

    /** A provider for {@code model} with the key from the environment. */
    public Gemini(String model) {
        this(builder(model));
    }

    public static Builder builder(String model) {
        return new Builder(model);
    }

    @Override
    public String url() {
        return baseUrl + "/models/" + Py.quote(model) + ":generateContent";
    }

    @Override
    protected Map<String, String> authHeaders() {
        Map<String, String> headers = new LinkedHashMap<>();
        headers.put("x-goog-api-key", apiKey());
        return headers;
    }

    private static String text(JsonNode content) {
        if (content != null && content.isTextual()) {
            return content.textValue();
        }
        if (content != null && content.isArray()) {  // OpenAI-style text parts
            List<String> parts = new ArrayList<>();
            for (JsonNode part : content) {
                if (part.isObject()) {
                    JsonNode text = part.get("text");
                    parts.add(text == null ? "" : text.asText());
                }
            }
            return String.join("\n", parts);
        }
        throw new IllegalArgumentException("Gemini messages need text content.");
    }

    @Override
    protected ObjectNode payload(ArrayNode messages, ArrayNode tools, ArrayNode history) {
        List<String> system = new ArrayList<>();
        system.add(systemPrompt);
        ArrayNode contents = Json.array();
        for (JsonNode message : Prompts.withHistory(messages, history)) {
            String role = Py.text(message.get("role"));
            if ("system".equals(role) || "developer".equals(role)) {
                system.add(text(message.get("content")));
            } else if ("user".equals(role) || "assistant".equals(role)) {
                ObjectNode content = contents.addObject();
                content.put("role", "assistant".equals(role) ? "model" : "user");
                content.putArray("parts").addObject().put("text", text(message.get("content")));
            } else {
                throw new IllegalArgumentException("Gemini accepts only user and assistant conversation messages.");
            }
        }
        ObjectNode payload = Json.object();
        payload.putObject("systemInstruction").putArray("parts").addObject().put("text", String.join("\n\n", system));
        payload.set("contents", contents);
        ArrayNode functions = Prompts.functionsOf(tools);
        if (!functions.isEmpty()) {
            ArrayNode declarations = payload.putArray("tools").addObject().putArray("functionDeclarations");
            for (JsonNode function : functions) {
                ObjectNode declaration = declarations.addObject();
                declaration.set("name", function.get("name").deepCopy());
                declaration.set("description", function.has("description") ? function.get("description").deepCopy()
                        : TextNode.valueOf(""));
                declaration.set("parametersJsonSchema", function.has("parameters")
                        ? function.get("parameters").deepCopy() : Anthropic.emptyParameters());
            }
            payload.putObject("toolConfig").putObject("functionCallingConfig").put("mode", "AUTO");
        }
        ObjectNode config = Json.object();
        if (maxOutputTokens != null && maxOutputTokens != 0) {
            config.put("maxOutputTokens", maxOutputTokens);
        }
        if (temperature != null) {
            config.set("temperature", Json.node(temperature));
        }
        if (thinkingBudget != null) {
            config.putObject("thinkingConfig").put("thinkingBudget", thinkingBudget);
        }
        if (!config.isEmpty()) {
            payload.set("generationConfig", config);
        }
        return payload;
    }

    @Override
    protected Decoded decode(ObjectNode data) {
        if (!Py.isNone(data.get("error"))) {
            throw new ProviderException("The response contained a provider error.");
        }
        JsonNode candidates = data.get("candidates");
        if (!Py.truthy(candidates)) {
            JsonNode feedback = data.get("promptFeedback");
            if (feedback != null && feedback.isObject() && Py.truthy(feedback.get("blockReason"))) {
                return Decoded.incomplete("prompt_filter");
            }
            throw new ProviderException("The Gemini response had no candidates.");
        }
        if (!candidates.isArray()) {
            throw new Py.ProtocolMismatch();
        }
        JsonNode candidate = candidates.get(0);
        if (!candidate.isObject()) {
            throw new Py.ProtocolMismatch();
        }
        String reason = Py.member(candidate.get("finishReason"));
        if ("MAX_TOKENS".equals(reason)) {
            return Decoded.incomplete("length");
        }
        if (reason != null && FILTERED.contains(reason)) {
            return Decoded.incomplete("content_filter");
        }
        if ("MALFORMED_FUNCTION_CALL".equals(reason) || "UNEXPECTED_TOOL_CALL".equals(reason)) {
            return Decoded.incomplete("malformed_call");
        }
        JsonNode content = Py.truthy(candidate.get("content")) ? candidate.get("content") : Json.object();
        if (!content.isObject()) {
            throw new Py.ProtocolMismatch();
        }
        List<JsonNode> parts = parts(content.get("parts"));
        List<Call> calls = new ArrayList<>();
        List<String> errors = new ArrayList<>();
        List<String> text = new ArrayList<>();
        for (JsonNode part : parts) {
            if (!part.isObject()) {
                throw new ProviderException("A Gemini content part was malformed.");
            }
            JsonNode call = part.get("functionCall");
            if (call != null && call.isObject()) {
                appendCall(calls, errors, call.get("name"), call.get("args"), false);
            } else if (part.get("text") != null && part.get("text").isTextual() && !part.get("text").textValue().isEmpty()
                    && !Py.truthy(part.get("thought"))) {
                text.add(part.get("text").textValue());
            }
        }
        return Decoded.complete(calls, errors, text);
    }

    /** The parts as Python iterates {@code parts or []}: a string or object yields non-part items. */
    private static List<JsonNode> parts(JsonNode parts) {
        List<JsonNode> result = new ArrayList<>();
        if (!Py.truthy(parts)) {
            return result;
        }
        if (parts.isArray()) {
            parts.forEach(result::add);
        } else if (parts.isObject()) {
            for (Iterator<String> it = parts.fieldNames(); it.hasNext(); ) {
                result.add(TextNode.valueOf(it.next()));
            }
        } else if (parts.isTextual()) {
            result.add(parts);
        } else {
            throw new Py.ProtocolMismatch();
        }
        return result;
    }

    @Override
    protected Map<String, Long> usage(ObjectNode data) {
        JsonNode raw = data.get("usageMetadata");
        if (raw == null || !raw.isObject()) {
            return new LinkedHashMap<>();
        }
        Long output = count(raw.get("candidatesTokenCount"));
        Long thoughts = count(raw.get("thoughtsTokenCount"));
        long thinking = thoughts == null ? 0 : thoughts;
        Map<String, Long> result = new LinkedHashMap<>();
        result.put("input_tokens", count(raw.get("promptTokenCount")));
        result.put("cached_input_tokens", count(raw.get("cachedContentTokenCount")));
        result.put("output_tokens", output == null && thinking == 0 ? null : (output == null ? 0 : output) + thinking);
        return withoutNulls(result);
    }

    /** Builds a {@link Gemini}. */
    public static final class Builder extends Provider.Builder<Builder, Gemini> {
        private Integer thinkingBudget;

        Builder(String model) {
            super(model);
        }

        /** {@code thinkingConfig.thinkingBudget}; none is sent by default. */
        public Builder thinkingBudget(Integer thinkingBudget) {
            this.thinkingBudget = thinkingBudget;
            return this;
        }

        @Override
        protected Builder self() {
            return this;
        }

        @Override
        public Gemini build() {
            return new Gemini(this);
        }
    }
}
