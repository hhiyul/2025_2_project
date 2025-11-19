package com.example.andro

import android.content.Context
import android.net.Uri
import android.webkit.MimeTypeMap
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import coil.compose.AsyncImage
import com.example.andro.network.InferenceResponse
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.RequestBody.Companion.toRequestBody
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale


// ===== Retrofit 인터페이스 & 데이터 클래스 =====
interface InferenceApi {
    @Multipart
    @POST("infer")
    suspend fun infer(@Part file: MultipartBody.Part): InferenceResponse
}


// ===== 업로드 + 추론 호출 =====
suspend fun uploadAndInfer(context: Context, uri: Uri): InferenceResponse =
    withContext(Dispatchers.IO) {
        val client = OkHttpClient.Builder()
            .addInterceptor { chain ->
                val newReq = chain.request().newBuilder()
                    .addHeader("X-API-Key", "fuck-key-123") // 🔑 FastAPI 인증 헤더
                    .build()
                chain.proceed(newReq)
            }
            .build()

        val retrofit = Retrofit.Builder()
            .baseUrl("https://uncially-engrossing-keeley.ngrok-free.dev/")   // PC에서 FastAPI가 돌고 있을 때
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build()

        val api = retrofit.create(InferenceApi::class.java)

        val cr = context.contentResolver
        val mime = cr.getType(uri) ?: run {
            val ext = MimeTypeMap.getFileExtensionFromUrl(uri.toString())
            MimeTypeMap.getSingleton().getMimeTypeFromExtension(ext) ?: "application/octet-stream"
        }

        val name = runCatching {
            cr.query(uri, null, null, null, null)?.use { c ->
                val nameIdx = c.getColumnIndex("_display_name")
                if (c.moveToFirst() && nameIdx >= 0) c.getString(nameIdx) else null
            }
        }.getOrNull() ?: ("upload." + (MimeTypeMap.getSingleton().getExtensionFromMimeType(mime) ?: "jpg"))

        val bytes = cr.openInputStream(uri)?.use { it.readBytes() }
            ?: error("이미지 열기 실패")

        val body = bytes.toRequestBody(mime.toMediaTypeOrNull())
        val part = MultipartBody.Part.createFormData("file", name, body)

        api.infer(part)
    }


@Composable
fun LoadingOverlay(isVisible: Boolean) {
    if (!isVisible) return

    // 점 0~3개 애니메이션
    var dotCount by remember { mutableStateOf(0) }

    LaunchedEffect(isVisible) {
        while (isVisible) {
            kotlinx.coroutines.delay(500)
            dotCount = (dotCount + 1) % 4
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black.copy(alpha = 0.3f)),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            CircularProgressIndicator()

            Text(
                text = "잠시만 기다려 주세요" + ".".repeat(dotCount),
                style = MaterialTheme.typography.bodyLarge,
                color = Color.White
            )
        }
    }
}

@Composable
fun ResultSheetContent(
    imageUri: Uri?,
    response: InferenceResponse
) {
    val confidencePercent = (response.confidence * 100).coerceIn(0.0, 100.0)
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // 위에 이미지
        Box(
            modifier = Modifier
                .size(200.dp)
                .clip(RoundedCornerShape(16.dp))
                .background(MaterialTheme.colorScheme.surfaceVariant),
            contentAlignment = Alignment.Center
        ) {
            if (imageUri != null) {
                AsyncImage(
                    model = imageUri,
                    contentDescription = "분석한 이미지",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop
                )
            } else {
                Text("이미지 없음")
            }
        }

        // 예측 결과
        Text(
            text = if (confidencePercent < 30.0) "추론 불가" else "예측 결과",
            style = MaterialTheme.typography.titleMedium
        )

        Text(
            text = if (confidencePercent < 30.0) "추론이 어려워요, 다른 각도나 사진으로 시도해 주세요" else response.prediction,
            style = if (confidencePercent < 30.0) MaterialTheme.typography.bodyMedium else MaterialTheme.typography.headlineMedium,
            color = if (confidencePercent < 30.0)
                MaterialTheme.colorScheme.error
            else
                MaterialTheme.colorScheme.onSurface
        )


        Spacer(modifier = Modifier.height(12.dp))
    }
}