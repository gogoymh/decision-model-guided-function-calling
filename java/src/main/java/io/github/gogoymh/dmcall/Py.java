package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/** Small emulations of the Python behavior the reference implementation relies on. */
final class Py {
    private Py() {
    }

    // Whitespace and case ------------------------------------------------------------------
    /** Python's {@code str.isspace} for one code point. */
    static boolean isSpace(int cp) {
        return Character.isWhitespace(cp) || Character.isSpaceChar(cp) || cp == 0x85;
    }

    /** Python's {@code str.strip()}. */
    static String strip(String s) {
        int start = 0;
        int end = s.length();
        while (start < end) {
            int cp = s.codePointAt(start);
            if (!isSpace(cp)) {
                break;
            }
            start += Character.charCount(cp);
        }
        while (end > start) {
            int cp = s.codePointBefore(end);
            if (!isSpace(cp)) {
                break;
            }
            end -= Character.charCount(cp);
        }
        return s.substring(start, end);
    }

    static String rstrip(String s, char c) {
        int end = s.length();
        while (end > 0 && s.charAt(end - 1) == c) {
            end--;
        }
        return s.substring(0, end);
    }

    /** Python's {@code str.isupper()}: at least one cased character and no lower or title case one. */
    static boolean isUpper(String s) {
        boolean cased = false;
        for (int i = 0; i < s.length(); ) {
            int cp = s.codePointAt(i);
            i += Character.charCount(cp);
            if (Character.isLowerCase(cp) || Character.isTitleCase(cp)) {
                return false;
            }
            if (Character.isUpperCase(cp)) {
                cased = true;
            }
        }
        return cased;
    }

    /** Python's ordering of {@code str}: by code point, not by UTF-16 unit. */
    static int compare(String a, String b) {
        int i = 0;
        int j = 0;
        while (i < a.length() && j < b.length()) {
            int x = a.codePointAt(i);
            int y = b.codePointAt(j);
            if (x != y) {
                return Integer.compare(x, y);
            }
            i += Character.charCount(x);
            j += Character.charCount(y);
        }
        return Integer.compare(a.length() - i, b.length() - j);
    }

    // Truthiness ---------------------------------------------------------------------------
    static boolean isNone(JsonNode v) {
        return v == null || v.isNull() || v.isMissingNode();
    }

    /** Python's {@code bool(value)} for a JSON value. */
    static boolean truthy(JsonNode v) {
        if (isNone(v)) {
            return false;
        }
        if (v.isBoolean()) {
            return v.booleanValue();
        }
        if (v.isIntegralNumber()) {
            return v.bigIntegerValue().signum() != 0;
        }
        if (v.isNumber()) {
            return v.doubleValue() != 0.0;
        }
        if (v.isTextual()) {
            return !v.textValue().isEmpty();
        }
        if (v.isContainerNode()) {
            return v.size() > 0;
        }
        return true;
    }

    /** The text of a string value, {@code null} otherwise. */
    static String text(JsonNode v) {
        return v != null && v.isTextual() ? v.textValue() : null;
    }

    /**
     * The value for a Python {@code value in {...}} test of strings. A list or an object is unhashable
     * there and raises {@code TypeError}, which the providers report as a protocol mismatch.
     */
    static String member(JsonNode v) {
        if (v != null && v.isContainerNode()) {
            throw new ProtocolMismatch();
        }
        return text(v);
    }

    /** Raised where the Python code would raise KeyError, TypeError, IndexError or AttributeError. */
    static final class ProtocolMismatch extends RuntimeException {
        ProtocolMismatch() {
            super("protocol mismatch", null, false, false);
        }
    }

    // repr() -------------------------------------------------------------------------------
    /** Python's {@code repr} of a JSON value as Python would hold it. */
    static String repr(JsonNode v) {
        StringBuilder sb = new StringBuilder();
        repr(sb, v);
        return sb.toString();
    }

    private static void repr(StringBuilder sb, JsonNode v) {
        if (isNone(v)) {
            sb.append("None");
        } else if (v.isBoolean()) {
            sb.append(v.booleanValue() ? "True" : "False");
        } else if (v.isIntegralNumber()) {
            sb.append(v.bigIntegerValue());
        } else if (v.isNumber()) {
            sb.append(floatRepr(v.doubleValue()));
        } else if (v.isTextual()) {
            sb.append(strRepr(v.textValue()));
        } else if (v.isArray()) {
            sb.append('[');
            for (int i = 0; i < v.size(); i++) {
                if (i > 0) {
                    sb.append(", ");
                }
                repr(sb, v.get(i));
            }
            sb.append(']');
        } else if (v.isObject()) {
            sb.append('{');
            boolean first = true;
            for (Iterator<Map.Entry<String, JsonNode>> it = v.fields(); it.hasNext(); ) {
                Map.Entry<String, JsonNode> entry = it.next();
                if (!first) {
                    sb.append(", ");
                }
                first = false;
                sb.append(strRepr(entry.getKey())).append(": ");
                repr(sb, entry.getValue());
            }
            sb.append('}');
        } else {
            sb.append(strRepr(v.asText()));
        }
    }

    /** Python's {@code repr} of a list of path components (strings and integers). */
    static String reprPath(List<Object> path) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < path.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            Object item = path.get(i);
            sb.append(item instanceof String ? strRepr((String) item) : String.valueOf(item));
        }
        return sb.append(']').toString();
    }

    /** Python's {@code repr} of a {@code str}. */
    static String strRepr(String s) {
        char quote = s.indexOf('\'') >= 0 && s.indexOf('"') < 0 ? '"' : '\'';
        StringBuilder sb = new StringBuilder(s.length() + 2).append(quote);
        for (int i = 0; i < s.length(); ) {
            int cp = s.codePointAt(i);
            i += Character.charCount(cp);
            if (cp == quote || cp == '\\') {
                sb.append('\\').appendCodePoint(cp);
            } else if (cp == '\t') {
                sb.append("\\t");
            } else if (cp == '\n') {
                sb.append("\\n");
            } else if (cp == '\r') {
                sb.append("\\r");
            } else if (cp < 0x20 || cp == 0x7f) {
                sb.append(String.format(Locale.ROOT, "\\x%02x", cp));
            } else if (cp < 0x7f || isPrintable(cp)) {
                sb.appendCodePoint(cp);
            } else if (cp <= 0xff) {
                sb.append(String.format(Locale.ROOT, "\\x%02x", cp));
            } else if (cp <= 0xffff) {
                sb.append(String.format(Locale.ROOT, "\\u%04x", cp));
            } else {
                sb.append(String.format(Locale.ROOT, "\\U%08x", cp));
            }
        }
        return sb.append(quote).toString();
    }

    private static boolean isPrintable(int cp) {
        switch (Character.getType(cp)) {
            case Character.CONTROL:
            case Character.FORMAT:
            case Character.SURROGATE:
            case Character.PRIVATE_USE:
            case Character.UNASSIGNED:
            case Character.LINE_SEPARATOR:
            case Character.PARAGRAPH_SEPARATOR:
            case Character.SPACE_SEPARATOR:
                return false;
            default:
                return true;
        }
    }

    // Floats -------------------------------------------------------------------------------
    /** Python's {@code repr(float)}: the shortest digits that round-trip, in Python's layout. */
    static String floatRepr(double d) {
        if (Double.isNaN(d)) {
            return "nan";
        }
        if (Double.isInfinite(d)) {
            return d > 0 ? "inf" : "-inf";
        }
        if (d == 0.0) {
            return Double.doubleToRawLongBits(d) < 0 ? "-0.0" : "0.0";
        }
        double magnitude = Math.abs(d);
        BigDecimal shortest = shortestDigits(magnitude).stripTrailingZeros();
        String digits = shortest.unscaledValue().toString();
        int decpt = digits.length() - shortest.scale();
        StringBuilder sb = new StringBuilder(24);
        if (d < 0) {
            sb.append('-');
        }
        if (decpt <= -4 || decpt > 16) {
            sb.append(digits.charAt(0));
            if (digits.length() > 1) {
                sb.append('.').append(digits, 1, digits.length());
            }
            int exponent = decpt - 1;
            sb.append('e').append(exponent < 0 ? '-' : '+');
            String e = Integer.toString(Math.abs(exponent));
            if (e.length() < 2) {
                sb.append('0');
            }
            sb.append(e);
        } else if (decpt <= 0) {
            sb.append("0.");
            sb.append("0".repeat(-decpt));
            sb.append(digits);
        } else if (decpt >= digits.length()) {
            sb.append(digits).append("0".repeat(decpt - digits.length())).append(".0");
        } else {
            sb.append(digits, 0, decpt).append('.').append(digits, decpt, digits.length());
        }
        return sb.toString();
    }

    /** Python's JSON form of a float: {@code repr}, or NaN, Infinity and -Infinity. */
    static String floatJson(double d) {
        if (Double.isNaN(d)) {
            return "NaN";
        }
        if (Double.isInfinite(d)) {
            return d > 0 ? "Infinity" : "-Infinity";
        }
        return floatRepr(d);
    }

    private static BigDecimal shortestDigits(double magnitude) {
        BigDecimal exact = new BigDecimal(magnitude);
        for (int precision = 1; precision <= 17; precision++) {
            BigDecimal nearest = exact.round(new MathContext(precision, RoundingMode.HALF_EVEN));
            if (Double.parseDouble(nearest.toString()) == magnitude) {
                return nearest;
            }
            // Near a power of two the round-trip interval is asymmetric; try both neighbours.
            BigDecimal down = exact.round(new MathContext(precision, RoundingMode.DOWN));
            if (Double.parseDouble(down.toString()) == magnitude) {
                return down;
            }
            BigDecimal up = exact.round(new MathContext(precision, RoundingMode.UP));
            if (Double.parseDouble(up.toString()) == magnitude) {
                return up;
            }
        }
        return exact.round(new MathContext(17, RoundingMode.HALF_EVEN));
    }

    private static final Pattern FLOAT = Pattern.compile(
            "[+-]?(?:\\d(?:_?\\d)*(?:\\.(?:\\d(?:_?\\d)*)?)?|\\.\\d(?:_?\\d)*)(?:[eE][+-]?\\d(?:_?\\d)*)?");
    private static final Pattern SPECIAL = Pattern.compile("([+-]?)(inf|infinity|nan)", Pattern.CASE_INSENSITIVE);

    /** Python's {@code float(text)}, or {@code null} where it raises ValueError. */
    static Double parseFloat(String text) {
        String t = strip(text);
        Matcher special = SPECIAL.matcher(t);
        if (special.matches()) {
            if (special.group(2).equalsIgnoreCase("nan")) {
                return Double.NaN;
            }
            return "-".equals(special.group(1)) ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
        }
        if (!FLOAT.matcher(t).matches()) {
            return null;
        }
        return Double.parseDouble(t.replace("_", ""));
    }

    // URLs and patterns --------------------------------------------------------------------
    /** Python's {@code urllib.parse.quote(text, safe='')}. */
    static String quote(String text) {
        StringBuilder sb = new StringBuilder();
        for (byte b : text.getBytes(StandardCharsets.UTF_8)) {
            int c = b & 0xff;
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
                    || c == '_' || c == '.' || c == '-' || c == '~') {
                sb.append((char) c);
            } else {
                sb.append('%').append(String.format(Locale.ROOT, "%02X", c));
            }
        }
        return sb.toString();
    }

    /** A Python regular expression as a Java pattern; named groups use Python's spelling. */
    static Pattern regex(String pattern) {
        String translated = pattern.replace("(?P<", "(?<").replaceAll("\\(\\?P=(\\w+)\\)", "\\\\k<$1>");
        try {
            return Pattern.compile(translated, Pattern.UNICODE_CHARACTER_CLASS);
        } catch (PatternSyntaxException e) {
            throw new IllegalArgumentException("Invalid regular expression.", e);
        }
    }

    /** Python's {@code urllib.parse.urlsplit}, enough for base URL checks. */
    record UrlParts(String scheme, String netloc, String path, String query, String fragment) {
        static UrlParts split(String url) {
            int start = 0;
            while (start < url.length() && url.charAt(start) <= ' ') {
                start++;
            }
            String rest = url.substring(start).replace("\t", "").replace("\r", "").replace("\n", "");
            String scheme = "";
            int colon = rest.indexOf(':');
            if (colon > 0 && isAsciiLetter(rest.charAt(0))) {
                boolean valid = true;
                for (int i = 0; i < colon; i++) {
                    char c = rest.charAt(i);
                    if (!(isAsciiLetter(c) || (c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.')) {
                        valid = false;
                        break;
                    }
                }
                if (valid) {
                    scheme = rest.substring(0, colon).toLowerCase(Locale.ROOT);
                    rest = rest.substring(colon + 1);
                }
            }
            String netloc = "";
            if (rest.startsWith("//")) {
                int end = rest.length();
                for (char c : new char[]{'/', '?', '#'}) {
                    int at = rest.indexOf(c, 2);
                    if (at >= 0) {
                        end = Math.min(end, at);
                    }
                }
                netloc = rest.substring(2, end);
                rest = rest.substring(end);
                if (netloc.contains("[") != netloc.contains("]")) {
                    throw new IllegalArgumentException("Invalid IPv6 URL");
                }
            }
            String fragment = "";
            int hash = rest.indexOf('#');
            if (hash >= 0) {
                fragment = rest.substring(hash + 1);
                rest = rest.substring(0, hash);
            }
            String query = "";
            int question = rest.indexOf('?');
            if (question >= 0) {
                query = rest.substring(question + 1);
                rest = rest.substring(0, question);
            }
            return new UrlParts(scheme, netloc, rest, query, fragment);
        }

        boolean hasUserInfo() {
            return netloc.contains("@");
        }

        String hostname() {
            String host = netloc.substring(netloc.lastIndexOf('@') + 1);
            int bracket = host.indexOf('[');
            if (bracket >= 0) {
                host = host.substring(bracket + 1);
                int close = host.indexOf(']');
                if (close >= 0) {
                    host = host.substring(0, close);
                }
            } else {
                int colon = host.indexOf(':');
                if (colon >= 0) {
                    host = host.substring(0, colon);
                }
            }
            return host.isEmpty() ? null : host.toLowerCase(Locale.ROOT);
        }

        private static boolean isAsciiLetter(char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        }
    }

    /**
     * The base URL rule shared by providers and decision models: an http(s) URL without
     * credentials, query or fragment, trailing slashes removed. Returns {@code null} if invalid.
     */
    static String baseUrl(String value) {
        String url = rstrip(strip(value == null ? "" : value), '/');
        UrlParts parts = UrlParts.split(url);
        if (!(parts.scheme().equals("http") || parts.scheme().equals("https")) || parts.netloc().isEmpty()
                || parts.hasUserInfo() || !parts.query().isEmpty() || !parts.fragment().isEmpty()) {
            return null;
        }
        return url;
    }
}
