package com.dhara.dhara_backend;

import org.springframework.http.*;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Arrays;

@RestController
@RequestMapping("/api/v1")
public class FileUploadController {

    private final RestTemplate restTemplate = new RestTemplate();
    private final String fastApiBaseUrl = "https://dhara-model-92062613767.asia-south1.run.app";

    @PostMapping("/upload")
    public ResponseEntity<String> uploadFiles(@RequestParam("files") MultipartFile[] files) {
        String uploadEndpoint = fastApiBaseUrl + "/upload-files";

        try {
            // Prepare the files and metadata
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            Arrays.stream(files).forEach(file -> {
                try {
                    // Create a FileSystemResource for each file
                    HttpHeaders headers = new HttpHeaders();
                    headers.setContentType(MediaType.MULTIPART_FORM_DATA);
                    headers.setContentDispositionFormData("files", file.getOriginalFilename());

                    // Add file content to the request body
                    HttpEntity<byte[]> fileEntity = new HttpEntity<>(file.getBytes(), headers);
                    body.add("files", fileEntity);
                } catch (IOException e) {
                    throw new RuntimeException("Error processing file: " + file.getOriginalFilename(), e);
                }
            });

            // Prepare the HTTP request
            HttpHeaders requestHeaders = new HttpHeaders();
            requestHeaders.setContentType(MediaType.MULTIPART_FORM_DATA);

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, requestHeaders);

            // Make the POST request to FastAPI
            ResponseEntity<String> response = restTemplate.postForEntity(uploadEndpoint, requestEntity, String.class);

            return ResponseEntity.status(response.getStatusCode()).body(response.getBody());

        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Error during file upload: " + e.getMessage());
        }
    }
}
