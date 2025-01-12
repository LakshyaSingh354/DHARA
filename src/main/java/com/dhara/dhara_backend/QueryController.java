package com.dhara.dhara_backend;

import org.springframework.http.HttpHeaders;
import java.util.Map;

import org.springframework.http.MediaType;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

@RestController
@RequestMapping("/api/v1")
public class QueryController {
    @PostMapping("/query")
    public ResponseEntity<String> sendQuery(@RequestBody Map<String, String> body) {
    String queryEndpoint = "https://dhara-model-92062613767.asia-south1.run.app/query";
    RestTemplate restTemplate = new RestTemplate();

    HttpHeaders headers = new HttpHeaders();
    headers.setContentType(MediaType.APPLICATION_JSON);

    HttpEntity<Map<String, String>> requestEntity = new HttpEntity<>(body, headers);

    try {
        ResponseEntity<String> response = restTemplate.exchange(
            queryEndpoint,
            HttpMethod.POST,
            requestEntity,
            String.class
        );
        return ResponseEntity.ok(response.getBody());
    } catch (Exception e) {
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
            .body("Error during query: " + e.getMessage());
    }
}

    
}
