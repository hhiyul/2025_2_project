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
import coil.compose.AsyncImage
import com.example.andro.network.InferenceResponse
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.RequestBody.Companion.toRequestBody
import retrofit2.Retrofit
import android.app.Activity
import android.content.pm.ActivityInfo
import android.content.res.Configuration
import androidx.compose.ui.platform.LocalContext
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

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

//네트워크 통신중 화면 전환 방지
@Composable
fun LockOrientationWhileLoading(isLoading: Boolean) {
    val context = LocalContext.current
    val activity = context as? Activity
    val savedOrientation = remember {
        mutableStateOf(ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED)
    }

    LaunchedEffect(isLoading) {
        if (activity == null) return@LaunchedEffect

        if (isLoading) {
            // 지금 설정을 저장해 둔다
            savedOrientation.value = activity.requestedOrientation

            // 아직 아무 고정이 없는 상태라면(UNSPECIFIED) → 현재 방향으로 잠그기
            if (activity.requestedOrientation == ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED) {
                val currentOrientation = activity.resources.configuration.orientation
                val lockOrientation =
                    if (currentOrientation == Configuration.ORIENTATION_LANDSCAPE)
                        ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
                    else
                        ActivityInfo.SCREEN_ORIENTATION_PORTRAIT

                activity.requestedOrientation = lockOrientation
            }
        } else {
            // 로딩 끝나면 다시 원래 설정으로 돌리기
            activity.requestedOrientation =
                if (savedOrientation.value == ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED)
                    ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED
                else
                    savedOrientation.value
        }
    }
}