package org.example.kt_backend.service.dto

data class InferenceResponse(
    val filename: String,
    val content_type: String?,
    val size_bytes: Int,
    val prediction: String
)
