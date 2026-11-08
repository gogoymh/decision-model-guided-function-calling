package io.github.gogoymh.dmcall;

/**
 * TypeSafe's hosted decision model. Key: {@code apiKey(...)} or {@code TYPESAFE_API_KEY}.
 *
 * <p>The default model is {@code jev-1.13.0}, the one used in the paper ({@code TYPESAFE_MODEL} or
 * {@code model(...)} changes it), and the default endpoint is {@code https://api.typesafe.ai}
 * ({@code TYPESAFE_BASE_URL} or {@code baseUrl(...)}).
 */
public final class Jev extends SystemOneDecisionModel {
    public static final String DEFAULT_MODEL = "jev-1.13.0";
    public static final String DEFAULT_BASE_URL = "https://api.typesafe.ai";
    /** The {@code source} of Jev's feedback records. */
    public static final String FEEDBACK_SOURCE = "jev_noul_and_deterministic_schema_validation";

    private Jev(Builder b, String apiKey) {
        super(firstNonEmpty(b.baseUrl, Env.raw("TYPESAFE_BASE_URL"), DEFAULT_BASE_URL),
                firstNonEmpty(b.model, Env.raw("TYPESAFE_MODEL"), DEFAULT_MODEL), apiKey, b, 60.0, 3, true);
    }

    /** Jev with the key from {@code TYPESAFE_API_KEY} and the paper's threshold (0.5). */
    public Jev() {
        this(builder());
    }

    /** Jev with this key and the paper's threshold (0.5). */
    public Jev(String apiKey) {
        this(builder().apiKey(apiKey));
    }

    private Jev(Builder b) {
        this(b, key(b.apiKey));
    }

    public static Builder builder() {
        return new Builder();
    }

    private static String key(String apiKey) {
        if (apiKey != null && !apiKey.isEmpty()) {
            return apiKey;
        }
        String fromEnv = Env.raw("TYPESAFE_API_KEY");
        String stripped = fromEnv == null ? "" : Py.strip(fromEnv);
        if (stripped.isEmpty()) {
            throw new IllegalArgumentException("Jev needs an API key: pass apiKey(...) or set TYPESAFE_API_KEY.");
        }
        return stripped;
    }

    private static String firstNonEmpty(String... values) {
        for (String value : values) {
            if (value != null && !value.isEmpty()) {
                return value;
            }
        }
        return null;
    }

    @Override
    protected String displayName() {
        return "Jev";
    }

    @Override
    protected String feedbackSource() {
        return FEEDBACK_SOURCE;
    }

    /** Builds a {@link Jev}. */
    public static final class Builder extends SystemOneDecisionModel.Builder<Builder> {
        private String apiKey;
        private String model;
        private String baseUrl;

        private Builder() {
        }

        public Builder apiKey(String apiKey) {
            this.apiKey = apiKey;
            return this;
        }

        /** The Jev model ({@code jev-1.13.0} by default). */
        public Builder model(String model) {
            this.model = model;
            return this;
        }

        /** The endpoint ({@code https://api.typesafe.ai} by default). */
        public Builder baseUrl(String baseUrl) {
            this.baseUrl = baseUrl;
            return this;
        }

        @Override
        protected Builder self() {
            return this;
        }

        public Jev build() {
            return new Jev(this);
        }
    }
}
