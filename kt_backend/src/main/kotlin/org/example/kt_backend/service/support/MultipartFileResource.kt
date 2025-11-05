package org.example.kt_backend.service.support

import org.springframework.core.io.ByteArrayResource
import org.springframework.web.multipart.MultipartFile

class MultipartFileResource(private val file: MultipartFile) : ByteArrayResource(file.bytes) {
    override fun getFilename(): String? = file.originalFilename ?: "upload"

    override fun contentLength(): Long = file.size
}
