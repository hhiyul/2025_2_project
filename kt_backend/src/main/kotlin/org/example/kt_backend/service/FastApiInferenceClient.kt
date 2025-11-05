package org.example.kt_backend.service

import org.example.kt_backend.config.FastApiClientProperties
import org.example.kt_backend.service.dto.InferenceResponse
import org.example.kt_backend.service.support.MultipartFileResource
import org.slf4j.LoggerFactory
import org.springframework.http.HttpEntity
import org.springframework.http.HttpHeaders
import org.springframework.http.HttpMethod
import org.springframework.http.HttpStatus
import org.springframework.http.MediaType
import org.springframework.stereotype.Service
import org.springframework.util.LinkedMultiValueMap
import org.springframework.web.client.RestClientException
import org.springframework.web.client.RestTemplate
import org.springframework.web.server.ResponseStatusException
import org.springframework.web.multipart.MultipartFile

@Service
class FastApiInferenceClient(
    private val restTemplate: RestTemplate,
    private val properties: FastApiClientProperties
) {

    private val logger = LoggerFactory.getLogger(javaClass)

    fun requestInference(file: MultipartFile): InferenceResponse {
        val targetUrl = properties.baseUrl.trimEnd('/') + properties.inferencePath
        val body = LinkedMultiValueMap<String, Any>()

        val fileHeaders = HttpHeaders().apply {
            contentType = MediaType.parseMediaType(file.contentType ?: MediaType.APPLICATION_OCTET_STREAM_VALUE)
            setContentDispositionFormData("file", file.originalFilename ?: "upload")
        }

        body.add("file", HttpEntity(MultipartFileResource(file), fileHeaders))

        val headers = HttpHeaders().apply {
            contentType = MediaType.MULTIPART_FORM_DATA
        }

        val requestEntity = HttpEntity(body, headers)

        return try {
            val response = restTemplate.exchange(
                targetUrl,
                HttpMethod.POST,
                requestEntity,
                InferenceResponse::class.java
            )
            response.body ?: throw ResponseStatusException(HttpStatus.BAD_GATEWAY, "Empty response from inference service")
        } catch (ex: RestClientException) {
            logger.error("Failed to call inference service", ex)
            throw ResponseStatusException(HttpStatus.BAD_GATEWAY, "Failed to call inference service", ex)
        }
    }
}
