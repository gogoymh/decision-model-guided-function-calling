package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A decision model's judgment of one candidate: the probabilities of yes, the verdicts (which
 * include the schema check) and the feedback record the generator reads if the candidate is
 * rejected. A probability is {@code null} for a question that was not asked.
 */
public final class Judgment implements JsonConvertible {
    private final String kind;
    private final Boolean functionPass;
    private final Boolean argumentsPass;
    private final boolean argumentsSkipped;
    private final Double functionProbability;
    private final Double argumentsProbability;
    private final Boolean accepted;
    private final ObjectNode feedback;
    private final List<SchemaError> schemaErrors;
    private final Map<String, Long> usage;
    private final String model;
    private final String requestedModel;
    private final double latencySeconds;
    private final double retryWaitSeconds;
    private final int apiCalls;

    private Judgment(Builder b) {
        this.kind = b.kind;
        this.functionPass = b.functionPass;
        this.argumentsPass = b.argumentsPass;
        this.argumentsSkipped = b.argumentsSkipped;
        this.functionProbability = b.functionProbability;
        this.argumentsProbability = b.argumentsProbability;
        this.accepted = b.accepted;
        this.feedback = b.feedback == null ? Json.object() : b.feedback.deepCopy();
        this.schemaErrors = List.copyOf(b.schemaErrors);
        this.usage = Collections.unmodifiableMap(new LinkedHashMap<>(b.usage));
        this.model = b.model;
        this.requestedModel = b.requestedModel;
        this.latencySeconds = b.latencySeconds;
        this.retryWaitSeconds = b.retryWaitSeconds;
        this.apiCalls = b.apiCalls;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** {@code null}, or {@code "initial_no_call_preserved"} for an initial no call returned unverified. */
    public String kind() {
        return kind;
    }

    /** V_f: the schema's function part passed and the function probability reached its threshold. */
    public Boolean functionPass() {
        return functionPass;
    }

    /** V_a, or {@code null} when the argument question was skipped (a no call). */
    public Boolean argumentsPass() {
        return argumentsPass;
    }

    public boolean argumentsSkipped() {
        return argumentsSkipped;
    }

    public Double functionProbability() {
        return functionProbability;
    }

    public Double argumentsProbability() {
        return argumentsProbability;
    }

    /** Both applicable checks passed; {@code null} for an unverified initial no call. */
    public Boolean accepted() {
        return accepted;
    }

    /** A copy of the feedback record. */
    public ObjectNode feedback() {
        return feedback.deepCopy();
    }

    public List<SchemaError> schemaErrors() {
        return schemaErrors;
    }

    /** {@code input_tokens} and {@code output_tokens}; a {@code null} count is unknown, never zero. */
    public Map<String, Long> usage() {
        return usage;
    }

    /** The model the decision model reported. */
    public String model() {
        return model;
    }

    public String requestedModel() {
        return requestedModel;
    }

    public double latencySeconds() {
        return latencySeconds;
    }

    public double retryWaitSeconds() {
        return retryWaitSeconds;
    }

    /** HTTP requests made, retries included. */
    public int apiCalls() {
        return apiCalls;
    }

    /** The fields of the Python reference's judgment dict. */
    @Override
    public JsonNode toJson() {
        ObjectNode node = Json.object();
        if (kind != null) {
            node.put("kind", kind);
        }
        node.put("function_pass", functionPass);
        node.put("arguments_pass", argumentsPass);
        node.put("arguments_skipped", argumentsSkipped);
        node.put("function_probability", functionProbability);
        node.put("arguments_probability", argumentsProbability);
        node.put("accepted", accepted);
        node.set("feedback", feedback.deepCopy());
        ArrayNode errors = node.putArray("schema_errors");
        schemaErrors.forEach(error -> errors.add(error.toJson()));
        node.set("usage", Proposal.usageJson(usage));
        node.put("model", model);
        node.put("requested_model", requestedModel);
        node.put("latency_s", latencySeconds);
        node.put("retry_wait_s", retryWaitSeconds);
        node.put("api_calls", apiCalls);
        return node;
    }

    @Override
    public String toString() {
        return "Judgment(accepted=" + accepted + ", function=" + functionProbability + ", arguments="
                + argumentsProbability + ")";
    }

    /** Builds a {@link Judgment}; custom decision models use it. */
    public static final class Builder {
        private String kind;
        private Boolean functionPass;
        private Boolean argumentsPass;
        private boolean argumentsSkipped;
        private Double functionProbability;
        private Double argumentsProbability;
        private Boolean accepted;
        private ObjectNode feedback;
        private List<SchemaError> schemaErrors = List.of();
        private Map<String, Long> usage = new LinkedHashMap<>();
        private String model;
        private String requestedModel;
        private double latencySeconds;
        private double retryWaitSeconds;
        private int apiCalls;

        private Builder() {
        }

        public Builder kind(String kind) {
            this.kind = kind;
            return this;
        }

        public Builder functionPass(Boolean functionPass) {
            this.functionPass = functionPass;
            return this;
        }

        public Builder argumentsPass(Boolean argumentsPass) {
            this.argumentsPass = argumentsPass;
            return this;
        }

        public Builder argumentsSkipped(boolean argumentsSkipped) {
            this.argumentsSkipped = argumentsSkipped;
            return this;
        }

        public Builder functionProbability(Double functionProbability) {
            this.functionProbability = functionProbability;
            return this;
        }

        public Builder argumentsProbability(Double argumentsProbability) {
            this.argumentsProbability = argumentsProbability;
            return this;
        }

        public Builder accepted(Boolean accepted) {
            this.accepted = accepted;
            return this;
        }

        public Builder feedback(ObjectNode feedback) {
            this.feedback = feedback;
            return this;
        }

        public Builder schemaErrors(List<SchemaError> schemaErrors) {
            this.schemaErrors = schemaErrors == null ? List.of() : schemaErrors;
            return this;
        }

        public Builder usage(Map<String, Long> usage) {
            this.usage = usage == null ? new LinkedHashMap<>() : new LinkedHashMap<>(usage);
            return this;
        }

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public Builder requestedModel(String requestedModel) {
            this.requestedModel = requestedModel;
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

        public Judgment build() {
            return new Judgment(this);
        }
    }
}
