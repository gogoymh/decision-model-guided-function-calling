package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * One candidate from a {@link Generator}.
 *
 * <p>{@link #calls()} is {@code null} when the output holds no complete proposal (cut off at the
 * token limit, filtered, refused or malformed; {@link #incomplete()} says why) and an empty list
 * for a decision to make no call. The two are never confused: an unreadable output is rejected and
 * repaired, while a readable initial no call may be returned as it is.
 */
public final class Proposal implements JsonConvertible {
    private final List<Call> calls;
    private final String content;
    private final List<String> parseErrors;
    private final String incomplete;
    private final String provider;
    private final String requestedModel;
    private final String model;
    private final Map<String, Long> usage;
    private final double latencySeconds;
    private final double retryWaitSeconds;
    private final int apiCalls;

    private Proposal(Builder b) {
        this.calls = Calls.copy(b.calls);
        this.content = b.content;
        this.parseErrors = List.copyOf(b.parseErrors);
        this.incomplete = b.incomplete;
        this.provider = b.provider;
        this.requestedModel = b.requestedModel;
        this.model = b.model;
        this.usage = Collections.unmodifiableMap(new LinkedHashMap<>(b.usage));
        this.latencySeconds = b.latencySeconds;
        this.retryWaitSeconds = b.retryWaitSeconds;
        this.apiCalls = b.apiCalls;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** A readable proposal: calls ({@code []} for no call) and optional text. */
    public static Proposal of(List<Call> calls, String content) {
        return builder().calls(calls).content(content).build();
    }

    /** An unreadable proposal, with one of the reasons in {@code spec/prompts.json} ("length", ...). */
    public static Proposal unreadable(String reason, String parseError) {
        return builder().calls(null).incomplete(reason).parseErrors(List.of(parseError)).build();
    }

    /** The calls, {@code []} for no call, {@code null} if unreadable. */
    public List<Call> calls() {
        return calls;
    }

    /** The model's text, or {@code null}. */
    public String content() {
        return content;
    }

    public List<String> parseErrors() {
        return parseErrors;
    }

    /** Why the output holds no complete proposal ({@code length}, {@code content_filter}, ...), or null. */
    public String incomplete() {
        return incomplete;
    }

    public String provider() {
        return provider;
    }

    public String requestedModel() {
        return requestedModel;
    }

    /** The model the provider reports, or {@code null}. */
    public String model() {
        return model;
    }

    /** Token counts (input_tokens, cached_input_tokens, output_tokens) the provider reported. */
    public Map<String, Long> usage() {
        return usage;
    }

    public double latencySeconds() {
        return latencySeconds;
    }

    public double retryWaitSeconds() {
        return retryWaitSeconds;
    }

    public int apiCalls() {
        return apiCalls;
    }

    /** A readable, completely parsed decision to make no call. */
    public boolean isReadableNoCall() {
        return calls != null && calls.isEmpty() && (incomplete == null || incomplete.isEmpty())
                && parseErrors.isEmpty();
    }

    /** The fields of the Python reference's proposal dict. */
    @Override
    public JsonNode toJson() {
        ObjectNode node = Json.object();
        node.set("calls", Calls.toJson(calls));
        node.put("content", content);
        ArrayNode errors = node.putArray("parse_errors");
        parseErrors.forEach(errors::add);
        node.put("incomplete", incomplete);
        node.put("provider", provider);
        node.put("requested_model", requestedModel);
        node.put("model", model);
        node.set("usage", usageJson(usage));
        node.put("latency_s", latencySeconds);
        node.put("retry_wait_s", retryWaitSeconds);
        node.put("api_calls", apiCalls);
        return node;
    }

    static ObjectNode usageJson(Map<String, Long> usage) {
        ObjectNode node = Json.object();
        usage.forEach((key, value) -> {
            if (value == null) {
                node.putNull(key);
            } else {
                node.put(key, value);
            }
        });
        return node;
    }

    @Override
    public String toString() {
        String decision = calls == null ? "unreadable (" + incomplete + ")" : calls.isEmpty() ? "no call"
                : calls.toString();
        return "Proposal(" + decision + ")";
    }

    /** Builds a {@link Proposal}; custom generators use it. */
    public static final class Builder {
        private List<Call> calls = List.of();
        private String content;
        private List<String> parseErrors = List.of();
        private String incomplete;
        private String provider;
        private String requestedModel;
        private String model;
        private Map<String, Long> usage = new LinkedHashMap<>();
        private double latencySeconds;
        private double retryWaitSeconds;
        private int apiCalls = 1;

        private Builder() {
        }

        public Builder calls(List<Call> calls) {
            this.calls = calls;
            return this;
        }

        public Builder content(String content) {
            this.content = content;
            return this;
        }

        public Builder parseErrors(List<String> parseErrors) {
            this.parseErrors = parseErrors == null ? List.of() : parseErrors;
            return this;
        }

        public Builder incomplete(String incomplete) {
            this.incomplete = incomplete;
            return this;
        }

        public Builder provider(String provider) {
            this.provider = provider;
            return this;
        }

        public Builder requestedModel(String requestedModel) {
            this.requestedModel = requestedModel;
            return this;
        }

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public Builder usage(Map<String, Long> usage) {
            this.usage = usage == null ? new LinkedHashMap<>() : new LinkedHashMap<>(usage);
            return this;
        }

        public Builder latencySeconds(double latencySeconds) {
            this.latencySeconds = latencySeconds;
            return this;
        }

        public Builder retryWaitSeconds(double retryWaitSeconds) {
            this.retryWaitSeconds = retryWaitSeconds;
            return this;
        }

        public Builder apiCalls(int apiCalls) {
            this.apiCalls = apiCalls;
            return this;
        }

        public Proposal build() {
            return new Proposal(this);
        }
    }
}
