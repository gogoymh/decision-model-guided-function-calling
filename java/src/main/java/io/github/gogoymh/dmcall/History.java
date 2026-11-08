package io.github.gogoymh.dmcall;

import java.util.Locale;

/** Which repair history the generator reads. */
public enum History {
    /** DM-Call: every earlier rejected candidate and its feedback. */
    CUMULATIVE("cumulative"),
    /** DM-Call-Latest: only the latest rejection, with its real round number. */
    LATEST("latest");

    private final String value;

    History(String value) {
        this.value = value;
    }

    /** The option's name in the Python reference and the spec. */
    public String value() {
        return value;
    }

    /** The member named {@code value} (as in the spec, for example {@code "cumulative"}). */
    public static History fromValue(String value) {
        for (History member : values()) {
            if (member.value.equals(value)) {
                return member;
            }
        }
        throw new IllegalArgumentException("history must be one of 'cumulative', 'latest'.");
    }

    @Override
    public String toString() {
        return value.toLowerCase(Locale.ROOT);
    }
}
