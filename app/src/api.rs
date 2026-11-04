//! A small HTTP/1.1 client for the local ghost-infer server: JSON requests, and the agent's server-sent
//! events read as they arrive. Localhost only, so no TLS and no HTTP crate.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use crate::json::Json;

/// Where the server is, and the token its tool endpoints ask for (the chat page carries it; this app reads
/// it from there, as any local program of the same user could).
#[derive(Debug, Clone)]
pub struct Api {
    pub addr: String,
    token: String,
}

/// A response's status and the reader positioned at its body.
struct Response {
    status: u16,
    chunked: bool,
    reader: BufReader<TcpStream>,
}

impl Api {
    /// Connects to `addr` (host:port) and picks up the tool token from the chat page.
    pub fn connect(addr: &str) -> Result<Api, String> {
        let mut api = Api { addr: addr.to_string(), token: String::new() };
        let page = api.body(api.send("GET", "/", None)?)?;
        let marker = "name=\"ghost-token\" content=\"";
        let at = page.find(marker).ok_or("this server has no tool token (is it an older ghost-infer?)")? + marker.len();
        api.token = page[at..].split('"').next().unwrap_or("").to_string();
        Ok(api)
    }

    fn send(&self, method: &str, path: &str, body: Option<&str>) -> Result<Response, String> {
        self.send_bytes(method, path, "application/json", body.unwrap_or("").as_bytes())
    }

    fn send_bytes(&self, method: &str, path: &str, content_type: &str, body: &[u8]) -> Result<Response, String> {
        let mut conn = TcpStream::connect(&self.addr).map_err(|e| format!("cannot reach ghost-infer at {}: {e}", self.addr))?;
        conn.set_read_timeout(Some(Duration::from_secs(1800))).ok();
        let _ = conn.set_nodelay(true);
        let head = format!(
            "{method} {path} HTTP/1.1\r\nHost: {}\r\nX-Ghost-Token: {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            self.addr,
            self.token,
            body.len()
        );
        conn.write_all(head.as_bytes()).and_then(|_| conn.write_all(body)).map_err(|e| e.to_string())?;
        let mut reader = BufReader::new(conn);
        let mut line = String::new();
        reader.read_line(&mut line).map_err(|e| e.to_string())?;
        let status = line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).ok_or("not an HTTP response")?;
        let mut chunked = false;
        loop {
            line.clear();
            if reader.read_line(&mut line).map_err(|e| e.to_string())? == 0 || line.trim().is_empty() {
                break;
            }
            let l = line.to_ascii_lowercase();
            if l.starts_with("transfer-encoding:") && l.contains("chunked") {
                chunked = true;
            }
        }
        Ok(Response { status, chunked, reader })
    }

    /// A whole response body as text (chunked or not).
    fn body(&self, mut r: Response) -> Result<String, String> {
        let mut out = Vec::new();
        if r.chunked {
            while let Some(chunk) = next_chunk(&mut r.reader)? {
                out.extend_from_slice(&chunk);
            }
        } else {
            r.reader.read_to_end(&mut out).map_err(|e| e.to_string())?;
        }
        let text = String::from_utf8_lossy(&out).into_owned();
        if r.status >= 400 {
            let msg = Json::parse(&text).ok().and_then(|j| j.get("error").and_then(|e| e.get("message").or(Some(e))).and_then(Json::as_str).map(String::from));
            return Err(msg.unwrap_or_else(|| format!("HTTP {}", r.status)));
        }
        Ok(text)
    }

    fn json(&self, method: &str, path: &str, body: Option<&Json>) -> Result<Json, String> {
        let text = self.body(self.send(method, path, body.map(|b| b.to_string()).as_deref())?)?;
        Json::parse(&text)
    }

    pub fn get(&self, path: &str) -> Result<Json, String> {
        self.json("GET", path, None)
    }

    pub fn post(&self, path: &str, body: &Json) -> Result<Json, String> {
        self.json("POST", path, Some(body))
    }

    pub fn delete(&self, path: &str) -> Result<Json, String> {
        self.json("DELETE", path, None)
    }

    /// POSTs raw bytes (a WAV file, for speech-to-text) and reads the JSON answer.
    pub fn post_bytes(&self, path: &str, content_type: &str, body: &[u8]) -> Result<Json, String> {
        let text = self.body(self.send_bytes("POST", path, content_type, body)?)?;
        Json::parse(&text)
    }

    /// POSTs `body` and hands each piece of a streamed binary answer (speech) to `on_chunk` as it arrives, until
    /// it ends or `on_chunk` returns false.
    pub fn stream_bytes(&self, path: &str, body: &Json, mut on_chunk: impl FnMut(&[u8]) -> bool) -> Result<(), String> {
        let mut r = self.send("POST", path, Some(&body.to_string()))?;
        if r.status >= 400 || !r.chunked {
            return Err(self.body(r).err().unwrap_or_else(|| "the server did not stream an answer".into()));
        }
        while let Some(chunk) = next_chunk(&mut r.reader)? {
            if !on_chunk(&chunk) {
                break;
            }
        }
        Ok(())
    }

    /// POSTs `body` and hands each server-sent event's JSON to `on_event` as it arrives, until the stream
    /// ends or `on_event` returns false (closing the connection stops the server's work).
    pub fn stream(&self, path: &str, body: &Json, mut on_event: impl FnMut(Json) -> bool) -> Result<(), String> {
        let mut r = self.send("POST", path, Some(&body.to_string()))?;
        if r.status >= 400 || !r.chunked {
            return Err(self.body(r).err().unwrap_or_else(|| "the server did not stream a reply".into()));
        }
        let mut pending = String::new();
        while let Some(chunk) = next_chunk(&mut r.reader)? {
            pending.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(end) = pending.find("\n\n") {
                let event: String = pending.drain(..end + 2).collect();
                for line in event.lines() {
                    let Some(data) = line.strip_prefix("data: ") else { continue };
                    if data == "[DONE]" {
                        return Ok(());
                    }
                    if let Ok(j) = Json::parse(data) {
                        if !on_event(j) {
                            return Ok(());
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// The next chunk of a chunked body, or None at its end.
fn next_chunk(r: &mut BufReader<TcpStream>) -> Result<Option<Vec<u8>>, String> {
    let mut line = String::new();
    if r.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
        return Ok(None);
    }
    let size = usize::from_str_radix(line.trim().split(';').next().unwrap_or("0"), 16).map_err(|_| format!("bad chunk size {line:?}"))?;
    if size == 0 {
        return Ok(None);
    }
    let mut buf = vec![0u8; size + 2]; // the data and its CRLF
    r.read_exact(&mut buf).map_err(|e| e.to_string())?;
    buf.truncate(size);
    Ok(Some(buf))
}
