package org.example.kt_backend

import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.context.properties.ConfigurationPropertiesScan
import org.springframework.boot.runApplication

@SpringBootApplication
@ConfigurationPropertiesScan
class KtBackendApplication

fun main(args: Array<String>) {
    runApplication<KtBackendApplication>(*args)
}
