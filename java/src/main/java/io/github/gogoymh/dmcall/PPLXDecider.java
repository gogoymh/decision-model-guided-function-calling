package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.util.Set;

/**
 * Perplexity Decider v1.1 27B, an open-weight decision model, served on your own machine.
 *
 * <p>{@code baseUrl} is the origin of a server running the checkpoint
 * ({@code perplexity-ai/pplx-decider-v1.1-27b} at revision {@code 14c9d901...}) with its own
 * native inference code, which serves {@code /v1/systemone}. A generic causal chat server is not
 * this model. Plain HTTP is accepted only for {@code localhost}; use an SSH tunnel or HTTPS
 * otherwise. Missing token usage stays unknown ({@code null}), never zero.
 *
 * <p>With {@code checkProvenance(true)} the client first reads {@code GET /provenance} and stops
 * unless the server reports the pinned checkpoint, revision, attention mode and calibration
 * temperature. The checkpoint's own server does not provide that route, so it is off by default.
 */
public final class PPLXDecider extends SystemOneDecisionModel {
    public static final String MODEL_ID = "perplexity-ai/pplx-decider-v1.1-27b";
    public static final String MODEL_REVISION = "14c9d90137962d300f3de95633f7b6535e80c5ce";
    public static final double CALIBRATION_TEMPERATURE = 1.0087417621345625;
    public static final int MAX_INPUT_TOKENS = 8192;
    /** The checkpoint's server reports this inherited name. */
    public static final String SERVER_MODEL = "autojev-qwen3.8-27b";
    /** The checkpoint's server accepts this as the request model. */
    public static final String WIRE_MODEL = "jev-latest";
    public static final String FEEDBACK_SOURCE = "perplexity_decider_noul_and_deterministic_schema_validation";

    private final boolean checkProvenance;
    private volatile ObjectNode provenance;

    private PPLXDecider(Builder b) {
        super(checkedOrigin(b.baseUrl), WIRE_MODEL, b.apiKey, b, 180.0, 0, false);
        this.checkProvenance = b.checkProvenance;
    }

    /** A decider at {@code baseUrl}, for example {@code http://127.0.0.1:18081}. */
    public PPLXDecider(String baseUrl) {
        this(builder(baseUrl));
    }

    public static Builder builder(String baseUrl) {
        return new Builder(baseUrl);
    }

    private static String checkedOrigin(String baseUrl) {
        Py.UrlParts parts = Py.UrlParts.split(baseUrl == null ? "" : baseUrl);
        if (!parts.path().isEmpty() && !parts.path().equals("/")) {
            throw new IllegalArgumentException("PPLXDecider baseUrl must be an origin without a path.");
        }
        String host = parts.hostname();
        if (parts.scheme().equals("http") && (host == null || !Set.of("localhost", "127.0.0.1", "::1").contains(host))) {
            throw new IllegalArgumentException("Use HTTPS, or an SSH tunnel to localhost, for a remote PPLX Decider.");
        }
        return baseUrl;
    }

    @Override
    protected String displayName() {
        return "PPLX Decider";
    }

    @Override
    protected String feedbackSource() {
        return FEEDBACK_SOURCE;
    }

    public boolean checkProvenance() {
        return checkProvenance;
    }

    /** The server's reported checkpoint, verified against the pinned one. */
    public ObjectNode provenance() {
        HttpTransport.Response response;
        try {
            response = transport().send(new HttpTransport.Request("GET", baseUrl() + "/provenance", requestHeaders(),
                    null, timeout()));
        } catch (IOException e) {
            throw new DecisionModelException("PPLX Decider provenance could not be reached.", "connection_failed");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new DecisionModelException("PPLX Decider provenance could not be reached.", "connection_failed");
        }
        if (response.status() != 200) {
            throw new DecisionModelException("PPLX Decider provenance is unavailable.", "provenance_unavailable",
                    response.status(), null, null);
        }
        JsonNode body;
        try {
            body = Json.parse(response.body());
        } catch (IllegalArgumentException e) {
            throw new DecisionModelException("PPLX Decider provenance was not valid JSON.", "provenance_invalid");
        }
        if (!body.isObject()) {
            throw new DecisionModelException("PPLX Decider provenance must be an object.", "provenance_invalid");
        }
        JsonNode temperature = body.get("temperature");
        JsonNode maxInput = body.get("max_input_tokens");
        boolean matches = MODEL_ID.equals(Py.text(body.get("model_id")))
                && MODEL_REVISION.equals(Py.text(body.get("revision")))
                && "noncausal_full_attention".equals(Py.text(body.get("native_attention")))
                && maxInput != null && maxInput.isNumber() && maxInput.doubleValue() == MAX_INPUT_TOKENS
                && temperature != null && temperature.isNumber()
                && Math.abs(temperature.doubleValue() - CALIBRATION_TEMPERATURE) <= 1e-12;
        if (!matches) {
            throw new DecisionModelException("The server does not run the pinned PPLX Decider checkpoint.",
                    "provenance_mismatch");
        }
        ObjectNode result = Json.object();
        result.put("model_id", MODEL_ID);
        result.put("revision", MODEL_REVISION);
        result.put("native_attention", "noncausal_full_attention");
        result.put("max_input_tokens", MAX_INPUT_TOKENS);
        result.put("temperature", temperature.doubleValue());
        return result;
    }

    @Override
    public Answer ask(ObjectNode state, ObjectNode questions) {
        if (checkProvenance && provenance == null) {
            provenance = provenance();
        }
        return super.ask(state, questions);
    }

    @Override
    protected DecisionModelException error(HttpTransport.Response response, int attempts) {
        DecisionModelException error = super.error(response, attempts);
        if (response.status() == 422) {
            try {
                JsonNode body = Json.parse(response.body());
                if (body.isObject()) {
                    JsonNode detail = body.get("detail");
                    String text = detail == null ? "" : detail.isTextual() ? detail.textValue() : Py.repr(detail);
                    if (text.contains("8192-token limit")) {
                        return error.withCode("context_limit_exceeded");
                    }
                }
            } catch (IllegalArgumentException ignored) {
                // Not JSON: keep the HTTP error.
            }
        }
        return error;
    }

    @Override
    protected ObjectNode checked(ObjectNode body) {
        String model = Py.text(body.get("model"));
        if (!MODEL_ID.equals(model) && !SERVER_MODEL.equals(model)) {
            throw new DecisionModelException("PPLX Decider response has an unexpected model ID.", "model_mismatch");
        }
        if (body.has("revision") && !MODEL_REVISION.equals(Py.text(body.get("revision")))) {
            throw new DecisionModelException("PPLX Decider response has an unexpected model revision.",
                    "model_mismatch");
        }
        ObjectNode result = body.deepCopy();
        result.put("model", MODEL_ID);
        if (!body.has("usage")) {
            // Missing usage is unknown, never zero.
            ObjectNode usage = result.putObject("usage");
            usage.putNull("input_tokens");
            usage.putNull("output_tokens");
        }
        return result;
    }

    @Override
    public String toString() {
        return "PPLXDecider(model=" + Py.strRepr(MODEL_ID) + ", revision=" + Py.strRepr(MODEL_REVISION.substring(0, 8))
                + ", baseUrl=" + Py.strRepr(baseUrl()) + ", functionThreshold=" + functionThreshold()
                + ", argumentsThreshold=" + argumentsThreshold() + ")";
    }

    /** Builds a {@link PPLXDecider}. Defaults: 180 second timeout, no HTTP retries. */
    public static final class Builder extends SystemOneDecisionModel.Builder<Builder> {
        private final String baseUrl;
        private String apiKey;
        private boolean checkProvenance;

        private Builder(String baseUrl) {
            this.baseUrl = baseUrl;
        }

        /** A key if your deployment wrapper requires one; the checkpoint's server needs none. */
        public Builder apiKey(String apiKey) {
            this.apiKey = apiKey;
            return this;
        }

        /** Check {@code GET /provenance} before the first request. */
        public Builder checkProvenance(boolean checkProvenance) {
            this.checkProvenance = checkProvenance;
            return this;
        }

        @Override
        protected Builder self() {
            return this;
        }

        public PPLXDecider build() {
            return new PPLXDecider(this);
        }
    }
}
