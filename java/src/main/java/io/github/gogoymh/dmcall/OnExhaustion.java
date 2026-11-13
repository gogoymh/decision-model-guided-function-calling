package io.github.gogoymh.dmcall;

import java.util.Locale;

/** Which candidate is final when none is accepted within the repair limit. */
public enum OnExhaustion {
    /** Return the first candidate (the paper's rule). */
    FIRST("first"),
    /** Return the last candidate. */
    LAST("last");

    private final String value;

    OnExhaustion(String value) {
        this.value = value;
    }

    /** The option's name in the Python reference and the spec. */
    public String value() {
        return value;
    }

    /** The member named {@code value} (as in the spec, for example {@code "first"}). */
    public static OnExhaustion fromValue(String value) {
        for (OnExhaustion member : values()) {
            if (member.value.equals(value)) {
                return member;
            }
        }
        throw new IllegalArgumentException("on_exhaustion must be one of 'first', 'last'.");
    }

    @Override
    public String toString() {
        return value.toLowerCase(Locale.ROOT);
    }
}
