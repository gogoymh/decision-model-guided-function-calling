//! Redirects must not forward provider credentials to another endpoint.
#![cfg(feature = "ureq")]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::Duration;

use dmcall::transport::UreqTransport;
use dmcall::{HttpRequest, HttpTransport, Method};

#[test]
fn redirect_is_returned_without_contacting_target() {
    let target = TcpListener::bind("127.0.0.1:0").unwrap();
    target.set_nonblocking(true).unwrap();
    let source = TcpListener::bind("127.0.0.1:0").unwrap();
    let source_url = format!("http://{}/start", source.local_addr().unwrap());
    let target_url = format!("http://{}/target", target.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        let (mut stream, _) = source.accept().unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
        let mut headers = Vec::new();
        let mut buffer = [0; 1024];
        loop {
            let read = stream.read(&mut buffer).unwrap();
            if read == 0 {
                break;
            }
            headers.extend_from_slice(&buffer[..read]);
            if headers.windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
        }
        stream.write_all(format!("HTTP/1.1 307 Temporary Redirect\r\nLocation: {target_url}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").as_bytes()).unwrap();
    });
    let request = HttpRequest {
        method: Method::Post,
        url: source_url,
        headers: vec![
            ("x-api-key".into(), "anthropic-secret".into()),
            ("api-key".into(), "azure-secret".into()),
            ("x-goog-api-key".into(), "gemini-secret".into()),
        ],
        body: b"{}".to_vec(),
        timeout: Duration::from_secs(3),
    };
    let response = UreqTransport::new(false).send(&request).unwrap();
    server.join().unwrap();
    assert_eq!(response.status, 307);
    assert_eq!(target.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock);
}
