package org.example.kt_backend.config

import org.springframework.boot.context.properties.ConfigurationProperties

@ConfigurationProperties(prefix = "fastapi.client")
data class FastApiClientProperties(
    var baseUrl: String = "http://localhost:8000",
    var inferencePath: String = "/infer"
)
