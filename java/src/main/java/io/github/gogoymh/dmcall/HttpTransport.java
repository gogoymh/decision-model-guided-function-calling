package io.github.gogoymh.dmcall;

import com.fasterxml.jackson.databind.JsonNode;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Sends one HTTP request. Generators and decision models use {@link #jdk()} by default; tests and
 * custom deployments inject their own. A failure to connect or a timeout is an {@link IOException}.
 */
@FunctionalInterface
public interface HttpTransport extends AutoCloseable {
    Response send(Request request) throws IOException, InterruptedException;

    @Override
    default void close() {
    }

    /** A transport over {@link java.net.http.HttpClient} (HTTP/1.1, system proxy settings). */
    static HttpTransport jdk() {
        return new JdkHttpTransport(Duration.ofSeconds(60), true);
    }

    /**
     * One request. {@link #toString()} leaves out headers and body, which may hold keys.
     *
     * @param method  {@code POST} or {@code GET}
     * @param url     the full URL
     * @param headers request headers in order
     * @param body    the body bytes ({@code null} for none)
     * @param timeout the time allowed for the response
     */
    record Request(String method, String url, Map<String, String> headers, byte[] body, Duration timeout) {
        public Request {
            headers = Collections.unmodifiableMap(new LinkedHashMap<>(headers));
        }

        /** A header's value by case-insensitive name, or {@code null}. */
        public String header(String name) {
            for (Map.Entry<String, String> entry : headers.entrySet()) {
                if (entry.getKey().equalsIgnoreCase(name)) {
                    return entry.getValue();
                }
            }
            return null;
        }

        /** The body as UTF-8 text. */
        public String bodyText() {
            return body == null ? "" : new String(body, StandardCharsets.UTF_8);
        }

        /** The body as JSON. */
        public JsonNode json() {
            return Json.parse(body);
        }

        @Override
        public String toString() {
            return "Request(" + method + " " + url + ")";
        }
    }

    /**
     * One response. {@link #toString()} leaves out headers and body.
     *
     * @param status  the HTTP status code
     * @param headers response headers; names compare case-insensitively
     * @param body    the body bytes
     */
    record Response(int status, Map<String, List<String>> headers, byte[] body) {
        public Response {
            TreeMap<String, List<String>> map = new TreeMap<>(String.CASE_INSENSITIVE_ORDER);
            if (headers != null) {
                for (Map.Entry<String, List<String>> entry : headers.entrySet()) {
                    map.merge(entry.getKey(), List.copyOf(entry.getValue()), (a, b) -> {
                        List<String> all = new ArrayList<>(a);
                        all.addAll(b);
                        return List.copyOf(all);
                    });
                }
            }
            headers = Collections.unmodifiableMap(map);
            body = body == null ? new byte[0] : body;
        }

        /** A response with a JSON body. */
        public static Response json(int status, JsonNode body) {
            return new Response(status, Map.of("content-type", List.of("application/json")),
                    Json.compact(body).getBytes(StandardCharsets.UTF_8));
        }

        /** A response with a body and single-valued headers. */
        public static Response of(int status, String body, Map<String, String> headers) {
            Map<String, List<String>> multi = new LinkedHashMap<>();
            if (headers != null) {
                headers.forEach((name, value) -> multi.put(name, List.of(value)));
            }
            return new Response(status, multi, body == null ? null : body.getBytes(StandardCharsets.UTF_8));
        }

        /** A header's value by case-insensitive name (several values joined by ", "), or {@code null}. */
        public String header(String name) {
            List<String> values = headers.get(name);
            return values == null || values.isEmpty() ? null : String.join(", ", values);
        }

        public boolean isSuccess() {
            return status >= 200 && status < 300;
        }

        @Override
        public String toString() {
            return "Response(" + status + ")";
        }
    }
}
