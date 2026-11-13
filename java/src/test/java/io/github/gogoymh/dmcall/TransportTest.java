package io.github.gogoymh.dmcall;

import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;

import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** The built-in transport never follows a redirect, so key headers stay with the configured host. */
class TransportTest {
    @Test
    void redirectIsReturnedWithoutContactingTheTarget() throws Exception {
        AtomicInteger sourceRequests = new AtomicInteger();
        AtomicInteger targetRequests = new AtomicInteger();
        HttpServer target = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        target.createContext("/", exchange -> {
            targetRequests.incrementAndGet();
            exchange.sendResponseHeaders(200, -1);
            exchange.close();
        });
        HttpServer source = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        source.createContext("/", exchange -> {
            sourceRequests.incrementAndGet();
            exchange.getRequestBody().readAllBytes();
            exchange.getResponseHeaders().add("Location",
                    "http://127.0.0.1:" + target.getAddress().getPort() + "/target");
            exchange.sendResponseHeaders(307, -1);
            exchange.close();
        });
        target.start();
        source.start();
        try (HttpTransport transport = HttpTransport.jdk()) {
            Map<String, String> headers = new LinkedHashMap<>();
            headers.put("x-api-key", "anthropic-secret");
            headers.put("api-key", "azure-secret");
            headers.put("x-goog-api-key", "gemini-secret");
            headers.put("Content-Type", "application/json");
            HttpTransport.Response response = transport.send(new HttpTransport.Request("POST",
                    "http://127.0.0.1:" + source.getAddress().getPort() + "/start", headers,
                    "{}".getBytes(StandardCharsets.UTF_8), Duration.ofSeconds(5)));
            assertEquals(307, response.status());
            assertEquals(1, sourceRequests.get());
            assertEquals(0, targetRequests.get());
        } finally {
            source.stop(0);
            target.stop(0);
        }
    }
}
