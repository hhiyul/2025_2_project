package com.example.andro

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.tooling.preview.PreviewScreenSizes
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import coil.compose.AsyncImage
import com.example.andro.network.InferenceResponse
import com.example.andro.ui.theme.AndroTheme
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import org.json.JSONObject

class MainActivity : ComponentActivity() {
    private val inferenceViewModel: InferenceViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            AndroTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    CameraAndGalleryScreen(viewModel = inferenceViewModel)
                }
            }
        }
    }
}

@PreviewScreenSizes
@Composable
fun CameraAndGalleryScreen(
    modifier: Modifier = Modifier,
    viewModel: InferenceViewModel = InferenceViewModel()
) {
    val context = LocalContext.current
    var selectedImageUri by remember { mutableStateOf<Uri?>(null) }
    var pendingCameraUri by remember { mutableStateOf<Uri?>(null) }
    var localMessage by remember { mutableStateOf<String?>(null) }
    val uiState = viewModel.uiState

    val takePictureLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) {
            selectedImageUri = pendingCameraUri
            localMessage = null
            viewModel.reset()
        } else {
            localMessage = "촬영이 취소되었거나 실패했습니다."
            pendingCameraUri?.let { uri ->
                runCatching { context.contentResolver.delete(uri, null, null) }
            }
        }
    }

    fun launchCamera() {
        val uri = runCatching { createImageUri(context) }
            .onFailure { throwable ->
                localMessage = throwable.localizedMessage ?: "카메라를 실행할 수 없습니다."
            }
            .getOrNull()

        if (uri != null) {
            pendingCameraUri = uri
            localMessage = null
            viewModel.reset()
            takePictureLauncher.launch(uri)
        }
    }

    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            launchCamera()
        } else {
            localMessage = "카메라 권한이 필요합니다."
        }
    }

    val pickVisualMediaLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        selectedImageUri = uri
        localMessage = null
        viewModel.reset()
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text(
            text = "사진을 촬영하거나 앨범에서 이미지를 선택하세요.",
            style = MaterialTheme.typography.titleMedium,
            textAlign = TextAlign.Center
        )

        Surface(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f, fill = true),
            tonalElevation = 2.dp,
            shape = MaterialTheme.shapes.medium
        ) {
            if (selectedImageUri != null) {
                AsyncImage(
                    model = selectedImageUri,
                    contentDescription = "선택된 이미지",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop
                )
            } else {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "이미지가 여기에 표시됩니다.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(24.dp)
                    )
                }
            }
        }

        val messageToShow = when (val state = uiState) {
            is InferenceUiState.Error -> state.message
            is InferenceUiState.Success -> formatInferenceResultAsJson(state.response)
            InferenceUiState.Idle -> "선택한 이미지를 확인한 뒤 추론을 실행하세요."
            InferenceUiState.Loading -> "추론을 실행 중입니다..."
        }

        val isError = localMessage != null || uiState is InferenceUiState.Error
        val textColor = if (isError) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurface
        val statusText = localMessage ?: messageToShow

        Text(
            text = statusText,
            color = textColor,
            style = MaterialTheme.typography.bodyMedium,
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth()
        )

        Button(
            enabled = uiState !is InferenceUiState.Loading,
            modifier = Modifier
                .fillMaxWidth()
                .height(60.dp)
                .padding(horizontal = 16.dp),
            onClick = {
                val uri = selectedImageUri
                if (uri == null) {
                    localMessage = "이미지를 먼저 선택하거나 촬영하세요."
                    return@Button
                }

                localMessage = null
                viewModel.inferImage(context, uri)
            }
        ) {
            Text(
                if (uiState is InferenceUiState.Loading) "추론 중..."
                else "추론하기"
            )
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Button(
                modifier = Modifier
                    .weight(1f)
                    .height(40.dp),
                onClick = {
                    if (
                        ContextCompat.checkSelfPermission(
                            context,
                            Manifest.permission.CAMERA
                        ) == PackageManager.PERMISSION_GRANTED
                    ) {
                        launchCamera()
                    } else {
                        cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                    }
                }
            ) {
                Text("카메라로 촬영")
            }

            Button(
                modifier = Modifier
                    .weight(1f)
                    .height(40.dp),
                onClick = {
                    pickVisualMediaLauncher.launch(
                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                    )
                }
            ) {
                Text("앨범에서 선택")
            }
        }

        Spacer(modifier = Modifier.height(8.dp))
    }
}

private fun createImageUri(context: Context): Uri? {
    val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
    val imageDir = File(context.cacheDir, "camera").apply {
        if (!exists()) mkdirs()
    }
    val imageFile = File.createTempFile("IMG_$timeStamp", ".jpg", imageDir)
    return FileProvider.getUriForFile(
        context,
        "${context.packageName}.fileprovider",
        imageFile
    )
}

fun formatInferenceResultAsJson(result: InferenceResponse): String {
    val json = JSONObject().apply {
        put("filename", result.filename)
        put("content_type", result.content_type)
        put("size_bytes", result.size_bytes)
        put("prediction", result.prediction)
        put("confidence", result.confidence)
    }
    return json.toString(2)
}

@Preview(showBackground = true)
@Composable
private fun CameraAndGalleryScreenPreview() {
    AndroTheme {
        CameraAndGalleryScreen()
    }
}
