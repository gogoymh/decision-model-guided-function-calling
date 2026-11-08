package io.github.gogoymh.dmcall;

import java.util.List;
import java.util.function.Function;

/** Environment lookups, replaceable in tests. */
final class Env {
    static volatile Function<String, String> source = System::getenv;

    private Env() {
    }

    /** The raw value, or {@code null}. */
    static String raw(String name) {
        return source.apply(name);
    }

    /** The first nonblank value among {@code names}, stripped, or {@code null}. */
    static String first(List<String> names) {
        for (String name : names) {
            String value = raw(name);
            if (value != null) {
                String stripped = Py.strip(value);
                if (!stripped.isEmpty()) {
                    return stripped;
                }
            }
        }
        return null;
    }
}
