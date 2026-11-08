package io.github.gogoymh.dmcall;

/** A sanitized generator failure. It never contains keys or response bodies. */
public class ProviderException extends RuntimeException {
    private final Integer statusCode;
    private final String requestId;
    private final Integer attempts;
    private final String providerCode;

    public ProviderException(String message) {
        this(message, null, null, null, null);
    }

    public ProviderException(String message, Integer statusCode, String requestId, Integer attempts,
                             String providerCode) {
        super(message);
        this.statusCode = statusCode;
        this.requestId = requestId;
        this.attempts = attempts;
        this.providerCode = providerCode;
    }

    /** The HTTP status, if a response arrived. */
    public Integer statusCode() {
        return statusCode;
    }

    /** A diagnostic request ID from the response headers, never a credential. */
    public String requestId() {
        return requestId;
    }

    /** HTTP attempts made, retries included. */
    public Integer attempts() {
        return attempts;
    }

    /** The provider's error code (for example {@code invalid_api_key}), if it gave a safe one. */
    public String providerCode() {
        return providerCode;
    }

    ProviderException withContext(Integer statusCode, String requestId, Integer attempts) {
        ProviderException copy = new ProviderException(getMessage(), statusCode, requestId, attempts, providerCode);
        copy.setStackTrace(getStackTrace());
        return copy;
    }
}
