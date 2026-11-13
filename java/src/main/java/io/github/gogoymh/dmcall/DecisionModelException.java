package io.github.gogoymh.dmcall;

/**
 * A decision-model transport or protocol failure. It is never a rejection, and it never holds keys
 * or bodies. {@link #code()} is one of {@code http_error}, {@code connection_failed},
 * {@code response_invalid}, {@code model_mismatch}, {@code context_limit_exceeded},
 * {@code provenance_unavailable}, {@code provenance_invalid}, {@code provenance_mismatch} or
 * {@code decision_model_error}.
 */
public class DecisionModelException extends RuntimeException {
    private final String code;
    private final Integer statusCode;
    private final String requestId;
    private final Integer attempts;

    public DecisionModelException(String message) {
        this(message, "decision_model_error", null, null, null);
    }

    public DecisionModelException(String message, String code) {
        this(message, code, null, null, null);
    }

    public DecisionModelException(String message, String code, Integer statusCode, String requestId,
                                  Integer attempts) {
        super(message);
        this.code = code == null ? "decision_model_error" : code;
        this.statusCode = statusCode;
        this.requestId = requestId;
        this.attempts = attempts;
    }

    public String code() {
        return code;
    }

    public Integer statusCode() {
        return statusCode;
    }

    public String requestId() {
        return requestId;
    }

    public Integer attempts() {
        return attempts;
    }

    DecisionModelException withCode(String newCode) {
        DecisionModelException copy = new DecisionModelException(getMessage(), newCode, statusCode, requestId,
                attempts);
        copy.setStackTrace(getStackTrace());
        return copy;
    }
}
