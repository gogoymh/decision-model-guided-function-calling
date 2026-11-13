package io.github.gogoymh.dmcall;

import java.util.Locale;

/** Which decision-model questions decide acceptance (the paper's DM-Call, DM-Call-Function and DM-Call-Argument). */
public enum Checks {
    /** DM-Call: both the function and the argument question. */
    BOTH("both"),
    /** DM-Call-Function: only the function question decides acceptance. */
    FUNCTION("function"),
    /** DM-Call-Argument: only the argument question decides acceptance. */
    ARGUMENTS("arguments");

    private final String value;

    Checks(String value) {
        this.value = value;
    }

    /** The option's name in the Python reference and the spec. */
    public String value() {
        return value;
    }

    /** The member named {@code value} (as in the spec, for example {@code "both"}). */
    public static Checks fromValue(String value) {
        for (Checks member : values()) {
            if (member.value.equals(value)) {
                return member;
            }
        }
        throw new IllegalArgumentException("checks must be one of 'both', 'function', 'arguments'.");
    }

    @Override
    public String toString() {
        return value.toLowerCase(Locale.ROOT);
    }
}
