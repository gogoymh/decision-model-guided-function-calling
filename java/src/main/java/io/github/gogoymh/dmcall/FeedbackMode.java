package io.github.gogoymh.dmcall;

import java.util.Locale;

/** How the decision model's scores reach the generator (the feedback representation). */
public enum FeedbackMode {
    /** Each probability keeps its value and threshold and gains a verbal label (the paper's default). */
    HYBRID("hybrid"),
    /** Probabilities and thresholds only. */
    NUMERIC("numeric"),
    /** Verbal labels only; probabilities and thresholds are removed. */
    TEXT("text");

    private final String value;

    FeedbackMode(String value) {
        this.value = value;
    }

    /** The option's name in the Python reference and the spec. */
    public String value() {
        return value;
    }

    /** The member named {@code value} (as in the spec, for example {@code "hybrid"}). */
    public static FeedbackMode fromValue(String value) {
        for (FeedbackMode member : values()) {
            if (member.value.equals(value)) {
                return member;
            }
        }
        throw new IllegalArgumentException("feedback must be one of 'hybrid', 'numeric', 'text'.");
    }

    @Override
    public String toString() {
        return value.toLowerCase(Locale.ROOT);
    }
}
