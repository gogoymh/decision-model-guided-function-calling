package io.github.gogoymh.dmcall;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Map;

/** {@link HttpTransport} over {@link HttpClient}, HTTP/1.1. */
final class JdkHttpTransport implements HttpTransport {
    private final HttpClient client;

    /**
     * @param connectTimeout the time allowed to connect
     * @param useProxy       whether to use the system proxy settings; {@code false} always connects
     *                       directly (the local PPLX Decider)
     */
    JdkHttpTransport(Duration connectTimeout, boolean useProxy) {
        HttpClient.Builder builder = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(connectTimeout)
                .followRedirects(HttpClient.Redirect.NEVER);
        if (!useProxy) {
            builder.proxy(HttpClient.Builder.NO_PROXY);
        }
        this.client = builder.build();
    }

    @Override
    public Response send(Request request) throws IOException, InterruptedException {
        HttpRequest.Builder builder = HttpRequest.newBuilder(URI.create(request.url()));
        if (request.timeout() != null) {
            builder.timeout(request.timeout());
        }
        for (Map.Entry<String, String> header : request.headers().entrySet()) {
            builder.header(header.getKey(), header.getValue());
        }
        if ("GET".equals(request.method())) {
            builder.GET();
        } else {
            builder.method(request.method(), request.body() == null ? HttpRequest.BodyPublishers.noBody()
                    : HttpRequest.BodyPublishers.ofByteArray(request.body()));
        }
        HttpResponse<byte[]> response = client.send(builder.build(), HttpResponse.BodyHandlers.ofByteArray());
        return new Response(response.statusCode(), response.headers().map(), response.body());
    }
}
