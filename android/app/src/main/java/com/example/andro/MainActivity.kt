package com.example.andro

import android.os.Bundle
import android.Manifest
import android.content.Context
import android.net.Uri
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.ui.platform.LocalConfiguration
import android.content.res.Configuration
import androidx.compose.ui.layout.ContentScale
import coil.compose.AsyncImage
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.rememberCoroutineScope
import kotlinx.coroutines.launch
import android.content.pm.PackageManager
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
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
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.PreviewScreenSizes
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.example.andro.ui.theme.AndroTheme
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            AndroTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    CameraAndGalleryScreen()
                }
            }
        }
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

@PreviewScreenSizes
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CameraAndGalleryScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current

    var selectedImageUriString by rememberSaveable { mutableStateOf<String?>(null) }
    val selectedImageUri: Uri? = selectedImageUriString?.let { Uri.parse(it) }

    var pendingCameraUriString by rememberSaveable { mutableStateOf<String?>(null) }
    val pendingCameraUri: Uri? = pendingCameraUriString?.let { Uri.parse(it)
    }
    var uiState by rememberSaveable {
        mutableStateOf<InferenceUiState>(InferenceUiState.Idle)
    }
    val scope = rememberCoroutineScope()

    // 🔹 화면 방향
    val configuration = LocalConfiguration.current
    val isLandscape = configuration.orientation == Configuration.ORIENTATION_LANDSCAPE

    // 결과 팝업용 상태
    val sheetState = rememberModalBottomSheetState(
        skipPartiallyExpanded = true
    )
    var showResultSheet by remember { mutableStateOf(false) }

    // Success 되면 자동으로 팝업 열기
    LaunchedEffect(uiState) {
        showResultSheet = uiState is InferenceUiState.Success
    }

    val takePictureLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) {
            selectedImageUriString = pendingCameraUri.toString()
            uiState = InferenceUiState.Idle
        } else {
            uiState = InferenceUiState.Error("촬영이 취소되었거나 실패했습니다.")
            pendingCameraUri?.let { uri ->
                runCatching { context.contentResolver.delete(uri, null, null) }
            }
        }
    }

    fun launchCamera() {
        val uri = runCatching { createImageUri(context) }
            .onFailure { throwable ->
                uiState = InferenceUiState.Error(
                    throwable.localizedMessage ?: "카메라를 실행할 수 없습니다."
                )
            }
            .getOrNull()

        if (uri != null) {
            pendingCameraUriString = uri.toString()
            uiState = InferenceUiState.Idle
            takePictureLauncher.launch(uri)
        }
    }

    // 카메라 권한 런처
    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            launchCamera()
        } else {
            uiState = InferenceUiState.Error("카메라 권한이 필요합니다.")
        }
    }

    // 앨범 선택 런처
    val pickVisualMediaLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        selectedImageUriString = uri.toString()
        uiState = InferenceUiState.Idle
    }
    LockOrientationWhileLoading(isLoading = uiState is InferenceUiState.Loading)
    // === 여기부터 레이아웃 ===
    Box(
        modifier = modifier
            .fillMaxSize()
    ) {
        if (!isLandscape) {
            // ================= 세로(기존 Column) =================
            Column(
                modifier = Modifier
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
                            contentScale = ContentScale.Fit
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

                // 추론 상태 안내
                when (val state = uiState) {
                    is InferenceUiState.Error -> {
                        Text(
                            text = state.message,
                            color = MaterialTheme.colorScheme.error,
                            style = MaterialTheme.typography.bodyMedium,
                            textAlign = TextAlign.Center,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }

                    is InferenceUiState.Success -> {
                        Text(
                            text = "예측 결과가 아래 팝업으로 표시됩니다.",
                            style = MaterialTheme.typography.bodySmall,
                            textAlign = TextAlign.Center,
                            modifier = Modifier.fillMaxWidth(),
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }

                    InferenceUiState.Idle -> {
                        Text(
                            text = "이미지를 확인한 뒤 추론하기를 눌러주세요.",
                            style = MaterialTheme.typography.bodySmall,
                            textAlign = TextAlign.Center,
                            modifier = Modifier.fillMaxWidth(),
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }

                    InferenceUiState.Loading -> {
                        Text(
                            text = "잠시만 기다려 주세요...",
                            style = MaterialTheme.typography.bodyMedium,
                            textAlign = TextAlign.Center,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                }

                // 추론하기
                Button(
                    enabled = uiState !is InferenceUiState.Loading,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(60.dp)
                        .padding(horizontal = 16.dp),
                    onClick = {
                        val uri = selectedImageUri
                        if (uri == null) {
                            uiState = InferenceUiState.Error("이미지를 먼저 선택하거나 촬영하세요.")
                            return@Button
                        }

                        scope.launch {
                            uiState = InferenceUiState.Loading
                            try {
                                val result = uploadAndInfer(context, uri)
                                uiState = InferenceUiState.Success(result)
                            } catch (e: Exception) {
                                uiState = InferenceUiState.Error(
                                    e.localizedMessage ?: "추론 요청 중 오류가 발생했습니다."
                                )
                            }
                        }
                    }
                ) {
                    Text(
                        if (uiState is InferenceUiState.Loading) "추론 중..."
                        else "추론하기"
                    )
                }

                // 하단 버튼들
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
        } else {
            // ================= 가로(왼쪽 이미지, 오른쪽 버튼들) =================
            Row(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(24.dp),
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // 왼쪽: 이미지 크게
                Surface(
                    modifier = Modifier
                        .weight(2f)
                        .fillMaxHeight(),
                    tonalElevation = 2.dp,
                    shape = MaterialTheme.shapes.medium
                ) {
                    if (selectedImageUri != null) {
                        AsyncImage(
                            model = selectedImageUri,
                            contentDescription = "선택된 이미지",
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Fit
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

                // 오른쪽: 안내 + 버튼들
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxHeight(),
                    verticalArrangement = Arrangement.spacedBy(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "사진을 촬영하거나 앨범에서 \n 이미지를 선택하세요.",
                        style = MaterialTheme.typography.titleMedium,
                        textAlign = TextAlign.Center
                    )

                    // 추론 상태 안내 (위와 동일 로직)
                    when (val state = uiState) {
                        is InferenceUiState.Error -> {
                            Text(
                                text = state.message,
                                color = MaterialTheme.colorScheme.error,
                                style = MaterialTheme.typography.bodyMedium,
                                textAlign = TextAlign.Center,
                                modifier = Modifier.fillMaxWidth()
                            )
                        }

                        is InferenceUiState.Success -> {
                            Text(
                                text = "예측 결과가 아래 팝업으로 표시됩니다.",
                                style = MaterialTheme.typography.bodySmall,
                                textAlign = TextAlign.Center,
                                modifier = Modifier.fillMaxWidth(),
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }

                        InferenceUiState.Idle -> {
                            Text(
                                text = "이미지를 확인한 뒤 추론하기를 눌러주세요.",
                                style = MaterialTheme.typography.bodySmall,
                                textAlign = TextAlign.Center,
                                modifier = Modifier.fillMaxWidth(),
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }

                        InferenceUiState.Loading -> {
                            Text(
                                text = "잠시만 기다려 주세요...",
                                style = MaterialTheme.typography.bodyMedium,
                                textAlign = TextAlign.Center,
                                modifier = Modifier.fillMaxWidth()
                            )
                        }
                    }

                    Button(
                        enabled = uiState !is InferenceUiState.Loading,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(60.dp),
                        onClick = {
                            val uri = selectedImageUri
                            if (uri == null) {
                                uiState = InferenceUiState.Error("이미지를 먼저 선택하거나 촬영하세요.")
                                return@Button
                            }

                            scope.launch {
                                uiState = InferenceUiState.Loading
                                try {
                                    val result = uploadAndInfer(context, uri)
                                    uiState = InferenceUiState.Success(result)
                                } catch (e: Exception) {
                                    uiState = InferenceUiState.Error(
                                        e.localizedMessage ?: "추론 요청 중 오류가 발생했습니다."
                                    )
                                }
                            }
                        }
                    ) {
                        Text(
                            if (uiState is InferenceUiState.Loading) "추론 중..."
                            else "추론하기"
                        )
                    }

                    Row(
                        modifier = Modifier.fillMaxWidth(),
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
                }
            }
        }

        // 🔥 로딩 오버레이: 화면 전체 덮기
        LoadingOverlay(isVisible = uiState is InferenceUiState.Loading)
    }

    if (showResultSheet) {
        val successState = uiState as? InferenceUiState.Success
        if (successState != null) {
            ModalBottomSheet(
                onDismissRequest = { showResultSheet = false },
                sheetState = sheetState
            ) {
                ResultSheetContent(
                    imageUri = selectedImageUri,
                    response = successState.response
                )
            }
        } else {
            showResultSheet = false
        }
    }
}
