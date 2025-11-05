package org.example.kt_backend.controller

import org.example.kt_backend.service.FastApiInferenceClient
import org.example.kt_backend.service.dto.InferenceResponse
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestPart
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.multipart.MultipartFile

@RestController
@RequestMapping("/api/images")
class ImageInferenceController(
    private val inferenceClient: FastApiInferenceClient
) {

    @PostMapping("/predict")
    fun predict(@RequestPart("file") file: MultipartFile): ResponseEntity<InferenceResponse> {
        val response = inferenceClient.requestInference(file)
        return ResponseEntity.ok(response)
    }
}
