package com.example.andro.network

import okhttp3.MultipartBody
import retrofit2.Response
import retrofit2.http.GET
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

// FastAPI /infer 응답 형식과 맞춰야 함
data class InferenceResponse(
    val filename: String,
    val prediction: String,
    val confidence: Float
)

interface ApiService {

    @GET("/health")
    suspend fun healthCheck(): Response<Map<String, String>>

    @Multipart
    @POST("/infer")
    suspend fun infer(
        @Part file: MultipartBody.Part
    ): Response<InferenceResponse>
}