#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace {

namespace py = pybind11;

constexpr int kWorkingSize = 64;
constexpr double kFeatureMarginRatio = 0.2;
constexpr double kExpandedFeatureMarginRatio = 0.15;
constexpr double kAppearanceErrorDecay = 0.08;
constexpr int kMinFingerprintInliers = 6;
constexpr double kMinTransformScale = 0.5;
constexpr double kMaxTransformScale = 2.0;
constexpr double kMinConfidentScale = 0.7;
constexpr double kMaxConfidentScale = 1.43;
constexpr int kMinConfidentInliers = 4;
constexpr double kMaxStrongInverseTransformError = 0.05;
constexpr double kMaxInverseTransformError = 0.15;
constexpr double kMaxScaleReciprocityLogError = 0.15;
constexpr double kMinScaleReciprocalRotation = 8.0;
constexpr double kInconsistentAppearanceExponent = 0.5;
constexpr double kFingerprintConfidenceExponent = 1.5;
constexpr double kStrongFingerprintConfidenceExponent = 2.0;
constexpr double kScaleReciprocalConfidenceExponent = 2.5;
constexpr double kKeypointDeduplicationRadius = 1.5;
constexpr int kSpatialVerificationGridSize = 4;
constexpr int kMinSpatialInlierCells = 4;
constexpr double kPartialSpatialEvidenceFactor = 0.6;
constexpr double kAlignedAppearanceMarginRatio = 0.1;
constexpr double kMaxAlignedAppearanceError = 0.13;
constexpr double kAlignedAppearanceErrorDecay = 0.05;
constexpr double kMinBackgroundEdgeContainment = 0.85;
constexpr double kMaxBackgroundErrorAllowance = 0.06;
constexpr double kMinSelfSufficientFingerprintScore = 0.8;
constexpr double kMissingStrictScopeFactor = 0.5;
constexpr double kStrictScopeWeight = 0.7;
constexpr double kMinStrongGeometricScore = 0.9;
constexpr double kMinStrongInlierSupport = 4.0;
constexpr double kFullStrongInlierSupport = 8.0;
constexpr double kMaxStrongInlierErrorAllowance = 0.105;
constexpr double kCentralConfirmationMarginRatio = 0.25;
constexpr int kMinCentralConfirmationInliers = 4;
constexpr int kFullCentralConfirmationInliers = 7;
constexpr double kCentralConfirmationConfidenceExponent = 2.0;
constexpr double kCentralContentMarginRatio = 0.25;
constexpr int kMinCentralContentEdgePixels = 32;
constexpr double kCentralContentEdgeWarning = 0.30;
constexpr double kCentralContentEdgeReject = 0.18;
constexpr double kCentralContentAppearanceEdgeIgnore = 0.50;
constexpr double kCentralContentErrorWarning = 0.095;
constexpr double kCentralContentErrorReject = 0.13;
constexpr double kMaxCentralContentPenalty = 0.8;

double clamp01(double value) {
    return std::max(0.0, std::min(1.0, value));
}

int bankers_round(double value) {
    return static_cast<int>(std::nearbyint(value));
}

// Match hashlib.blake2b(..., digest_size=16) so RANSAC input ordering remains
// deterministic and picture_match(a, b) is identical to picture_match(b, a).
std::uint64_t rotate_right(std::uint64_t value, unsigned int bits) {
    return (value >> bits) | (value << (64U - bits));
}

std::uint64_t load64_le(const unsigned char* source) {
    std::uint64_t value = 0;
    for (unsigned int index = 0; index < 8; ++index) {
        value |= static_cast<std::uint64_t>(source[index]) << (8U * index);
    }
    return value;
}

void store64_le(unsigned char* destination, std::uint64_t value) {
    for (unsigned int index = 0; index < 8; ++index) {
        destination[index] = static_cast<unsigned char>(value >> (8U * index));
    }
}

void blake2b_compress(
    std::array<std::uint64_t, 8>& state,
    const unsigned char block[128],
    std::uint64_t byte_count,
    bool final_block
) {
    static constexpr std::array<std::uint64_t, 8> initialization = {
        UINT64_C(0x6a09e667f3bcc908), UINT64_C(0xbb67ae8584caa73b),
        UINT64_C(0x3c6ef372fe94f82b), UINT64_C(0xa54ff53a5f1d36f1),
        UINT64_C(0x510e527fade682d1), UINT64_C(0x9b05688c2b3e6c1f),
        UINT64_C(0x1f83d9abfb41bd6b), UINT64_C(0x5be0cd19137e2179),
    };
    static constexpr unsigned char schedule[12][16] = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
        {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
        {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
        {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
        {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
        {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
        {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
        {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
        {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
        {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
        {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    };
    std::array<std::uint64_t, 16> message{};
    std::array<std::uint64_t, 16> work{};
    for (int index = 0; index < 16; ++index) {
        message[index] = load64_le(block + index * 8);
    }
    for (int index = 0; index < 8; ++index) {
        work[index] = state[index];
        work[index + 8] = initialization[index];
    }
    work[12] ^= byte_count;
    if (final_block) {
        work[14] = ~work[14];
    }

    auto mix = [&](int a, int b, int c, int d, std::uint64_t x, std::uint64_t y) {
        work[a] = work[a] + work[b] + x;
        work[d] = rotate_right(work[d] ^ work[a], 32);
        work[c] += work[d];
        work[b] = rotate_right(work[b] ^ work[c], 24);
        work[a] = work[a] + work[b] + y;
        work[d] = rotate_right(work[d] ^ work[a], 16);
        work[c] += work[d];
        work[b] = rotate_right(work[b] ^ work[c], 63);
    };
    for (int round = 0; round < 12; ++round) {
        const unsigned char* permutation = schedule[round];
        mix(0, 4, 8, 12, message[permutation[0]], message[permutation[1]]);
        mix(1, 5, 9, 13, message[permutation[2]], message[permutation[3]]);
        mix(2, 6, 10, 14, message[permutation[4]], message[permutation[5]]);
        mix(3, 7, 11, 15, message[permutation[6]], message[permutation[7]]);
        mix(0, 5, 10, 15, message[permutation[8]], message[permutation[9]]);
        mix(1, 6, 11, 12, message[permutation[10]], message[permutation[11]]);
        mix(2, 7, 8, 13, message[permutation[12]], message[permutation[13]]);
        mix(3, 4, 9, 14, message[permutation[14]], message[permutation[15]]);
    }
    for (int index = 0; index < 8; ++index) {
        state[index] ^= work[index] ^ work[index + 8];
    }
}

std::array<unsigned char, 16> content_digest(const cv::Mat& image) {
    static constexpr std::array<std::uint64_t, 8> initialization = {
        UINT64_C(0x6a09e667f3bcc908), UINT64_C(0xbb67ae8584caa73b),
        UINT64_C(0x3c6ef372fe94f82b), UINT64_C(0xa54ff53a5f1d36f1),
        UINT64_C(0x510e527fade682d1), UINT64_C(0x9b05688c2b3e6c1f),
        UINT64_C(0x1f83d9abfb41bd6b), UINT64_C(0x5be0cd19137e2179),
    };
    cv::Mat contiguous = image.isContinuous() ? image : image.clone();
    const unsigned char* input = contiguous.data;
    std::size_t remaining = contiguous.total() * contiguous.elemSize();
    std::uint64_t processed = 0;
    std::array<std::uint64_t, 8> state = initialization;
    state[0] ^= UINT64_C(0x01010000) ^ 16U;
    while (remaining > 128) {
        processed += 128;
        blake2b_compress(state, input, processed, false);
        input += 128;
        remaining -= 128;
    }
    std::array<unsigned char, 128> final_block{};
    std::memcpy(final_block.data(), input, remaining);
    processed += remaining;
    blake2b_compress(state, final_block.data(), processed, true);

    std::array<unsigned char, 16> digest{};
    store64_le(digest.data(), state[0]);
    store64_le(digest.data() + 8, state[1]);
    return digest;
}

cv::Mat image_from_array(const py::array& array, const char* name) {
    if (!array.dtype().is(py::dtype::of<unsigned char>())) {
        throw std::invalid_argument(std::string(name) + " must contain uint8 pixels");
    }
    if ((array.flags() & py::array::c_style) == 0) {
        throw std::invalid_argument(std::string(name) + " must be C-contiguous");
    }

    const py::buffer_info view = array.request();
    if (view.ndim != 2 && view.ndim != 3) {
        throw std::invalid_argument(std::string(name) + " must have 2 or 3 dimensions");
    }
    if (view.shape[0] <= 0 || view.shape[1] <= 0
        || view.shape[0] > std::numeric_limits<int>::max()
        || view.shape[1] > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(name) + " has invalid dimensions");
    }

    const int channels = view.ndim == 2 ? 1 : static_cast<int>(view.shape[2]);
    if (channels != 1 && channels != 3 && channels != 4) {
        throw std::invalid_argument(std::string(name) + " must be grayscale, RGB, or RGBA");
    }
    cv::Mat wrapped(
        static_cast<int>(view.shape[0]),
        static_cast<int>(view.shape[1]),
        CV_MAKETYPE(CV_8U, channels),
        view.ptr,
        static_cast<std::size_t>(view.strides[0])
    );
    return wrapped.clone();
}

py::array image_to_array(const cv::Mat& image, bool preserve_channel_axis) {
    std::vector<py::ssize_t> shape = {
        static_cast<py::ssize_t>(image.rows),
        static_cast<py::ssize_t>(image.cols),
    };
    if (image.channels() > 1 || preserve_channel_axis) {
        shape.push_back(static_cast<py::ssize_t>(image.channels()));
    }
    py::array_t<unsigned char> result(shape);
    py::buffer_info output = result.request();
    const std::size_t row_bytes = static_cast<std::size_t>(
        image.cols * image.channels()
    );
    for (int row = 0; row < image.rows; ++row) {
        std::memcpy(
            static_cast<unsigned char*>(output.ptr) + row * row_bytes,
            image.ptr<unsigned char>(row),
            row_bytes
        );
    }
    return result;
}

cv::Mat to_grayscale(const cv::Mat& image) {
    if (image.channels() == 1) {
        return image;
    }
    cv::Mat gray;
    cv::cvtColor(
        image,
        gray,
        image.channels() == 3 ? cv::COLOR_RGB2GRAY : cv::COLOR_RGBA2GRAY
    );
    return gray;
}

cv::Size normalized_size(const cv::Mat& image) {
    const double scale = std::min(
        1.0,
        kWorkingSize / static_cast<double>(std::min(image.rows, image.cols))
    );
    return {
        std::max(1, bankers_round(image.cols * scale)),
        std::max(1, bankers_round(image.rows * scale)),
    };
}

cv::Scalar border_median(const cv::Mat& image) {
    cv::Scalar result{};
    const int channels = image.channels();
    const std::size_t count = static_cast<std::size_t>(2 * image.cols + 2 * image.rows);
    for (int channel = 0; channel < channels; ++channel) {
        std::vector<unsigned char> values;
        values.reserve(count);
        for (int x = 0; x < image.cols; ++x) {
            values.push_back(image.ptr<unsigned char>(0)[x * channels + channel]);
        }
        for (int x = 0; x < image.cols; ++x) {
            values.push_back(image.ptr<unsigned char>(image.rows - 1)[x * channels + channel]);
        }
        for (int y = 0; y < image.rows; ++y) {
            values.push_back(image.ptr<unsigned char>(y)[channel]);
        }
        for (int y = 0; y < image.rows; ++y) {
            values.push_back(image.ptr<unsigned char>(y)[(image.cols - 1) * channels + channel]);
        }
        std::sort(values.begin(), values.end());
        const std::size_t middle = values.size() / 2;
        double median = values[middle];
        if (values.size() % 2 == 0) {
            median = (static_cast<double>(values[middle - 1]) + values[middle]) / 2.0;
        }
        result[channel] = static_cast<unsigned char>(median);
    }
    return result;
}

cv::Mat pad_on_working_canvas(const cv::Mat& image) {
    if (image.rows > kWorkingSize || image.cols > kWorkingSize) {
        throw std::invalid_argument("working tile exceeds 64 x 64");
    }
    if (image.rows == kWorkingSize && image.cols == kWorkingSize) {
        return image;
    }
    cv::Mat canvas(kWorkingSize, kWorkingSize, image.type(), border_median(image));
    const int top = (kWorkingSize - image.rows) / 2;
    const int left = (kWorkingSize - image.cols) / 2;
    image.copyTo(canvas(cv::Rect(left, top, image.cols, image.rows)));
    return canvas;
}

std::vector<cv::Mat> split_working_tiles(const cv::Mat& image) {
    if (image.rows <= kWorkingSize && image.cols <= kWorkingSize) {
        return {pad_on_working_canvas(image)};
    }
    const bool horizontal = image.cols > image.rows;
    const int long_edge = std::max(image.rows, image.cols);
    const int tile_count = (long_edge + kWorkingSize - 1) / kWorkingSize;
    std::vector<cv::Mat> result;
    result.reserve(tile_count);
    for (int index = 0; index < tile_count; ++index) {
        const double position = tile_count == 1
            ? 0.0
            : index * (long_edge - kWorkingSize) / static_cast<double>(tile_count - 1);
        const int start = bankers_round(position);
        const cv::Mat tile = horizontal
            ? image(cv::Rect(start, 0, kWorkingSize, image.rows))
            : image(cv::Rect(0, start, image.cols, kWorkingSize));
        result.push_back(pad_on_working_canvas(tile));
    }
    return result;
}

cv::Mat image_edges_from_gray(const cv::Mat& gray) {
    cv::Mat blurred;
    cv::Mat edges;
    cv::GaussianBlur(gray, blurred, {3, 3}, 0.0);
    cv::Canny(blurred, edges, 50.0, 150.0);
    cv::threshold(edges, edges, 0.0, 1.0, cv::THRESH_BINARY);
    return edges;
}

cv::Mat gradient_magnitude(const cv::Mat& image) {
    const cv::Mat gray = to_grayscale(image);
    cv::Mat gradient_x;
    cv::Mat gradient_y;
    cv::Mat magnitude;
    cv::Sobel(gray, gradient_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gradient_y, CV_32F, 0, 1, 3);
    cv::magnitude(gradient_x, gradient_y, magnitude);

    cv::Mat normalized_float;
    cv::normalize(magnitude, normalized_float, 0.0, 255.0, cv::NORM_MINMAX);
    cv::Mat normalized(magnitude.size(), CV_8U, cv::Scalar(0));
    for (int y = 0; y < normalized_float.rows; ++y) {
        const float* source = normalized_float.ptr<float>(y);
        unsigned char* destination = normalized.ptr<unsigned char>(y);
        for (int x = 0; x < normalized_float.cols; ++x) {
            destination[x] = static_cast<unsigned char>(
                std::max(0.0f, std::min(255.0f, source[x]))
            );
        }
    }
    return normalized;
}

double appearance_score(const cv::Mat& image_a, const cv::Mat& image_b) {
    const int margin_y = bankers_round(image_a.rows * kFeatureMarginRatio);
    const int margin_x = bankers_round(image_a.cols * kFeatureMarginRatio);
    const cv::Rect central(
        margin_x,
        margin_y,
        image_a.cols - 2 * margin_x,
        image_a.rows - 2 * margin_y
    );
    const cv::Mat a = image_a(central);
    const cv::Mat b = image_b(central);
    double error_sum = 0.0;
    const std::size_t row_bytes = static_cast<std::size_t>(a.cols * a.channels());
    for (int y = 0; y < a.rows; ++y) {
        const unsigned char* row_a = a.ptr<unsigned char>(y);
        const unsigned char* row_b = b.ptr<unsigned char>(y);
        for (std::size_t index = 0; index < row_bytes; ++index) {
            error_sum += std::abs(static_cast<int>(row_a[index]) - static_cast<int>(row_b[index]));
        }
    }
    const double normalized_error = error_sum / (a.total() * a.channels() * 255.0);
    return std::exp(-normalized_error / kAppearanceErrorDecay);
}

unsigned int popcount64(std::uint64_t value) {
#if defined(_MSC_VER) && defined(_M_X64)
    return static_cast<unsigned int>(__popcnt64(value));
#else
    value -= (value >> 1) & UINT64_C(0x5555555555555555);
    value = (value & UINT64_C(0x3333333333333333)) + ((value >> 2) & UINT64_C(0x3333333333333333));
    value = (value + (value >> 4)) & UINT64_C(0x0f0f0f0f0f0f0f0f);
    return static_cast<unsigned int>((value * UINT64_C(0x0101010101010101)) >> 56);
#endif
}

std::array<std::uint64_t, kWorkingSize> edge_rows(const cv::Mat& edges) {
    std::array<std::uint64_t, kWorkingSize> rows{};
    for (int y = 0; y < kWorkingSize; ++y) {
        const unsigned char* pixels = edges.ptr<unsigned char>(y);
        std::uint64_t row = 0;
        for (int x = 0; x < kWorkingSize; ++x) {
            if (pixels[x] != 0) {
                row |= UINT64_C(1) << x;
            }
        }
        rows[y] = row;
    }
    return rows;
}

unsigned int edge_count(const std::array<std::uint64_t, kWorkingSize>& rows) {
    unsigned int count = 0;
    for (const std::uint64_t row : rows) {
        count += popcount64(row);
    }
    return count;
}

double shifted_f1(
    const std::array<std::uint64_t, kWorkingSize>& fixed,
    unsigned int fixed_count,
    const std::array<std::uint64_t, kWorkingSize>& moving,
    int dx,
    int dy
) {
    unsigned int moved_count = 0;
    unsigned int intersection = 0;
    for (int y = 0; y < kWorkingSize; ++y) {
        const int source_y = y - dy;
        if (source_y < 0 || source_y >= kWorkingSize) {
            continue;
        }
        const std::uint64_t shifted = dx >= 0
            ? moving[source_y] << dx
            : moving[source_y] >> (-dx);
        moved_count += popcount64(shifted);
        intersection += popcount64(fixed[y] & shifted);
    }
    return fixed_count + moved_count == 0
        ? 1.0
        : 2.0 * intersection / static_cast<double>(fixed_count + moved_count);
}

double best_shifted_f1(
    const std::array<std::uint64_t, kWorkingSize>& fixed,
    unsigned int fixed_count,
    const std::array<std::uint64_t, kWorkingSize>& moving
) {
    double best = 0.0;
    for (int dx = -5; dx <= 5; ++dx) {
        for (int dy = -5; dy <= 5; ++dy) {
            best = std::max(best, shifted_f1(fixed, fixed_count, moving, dx, dy));
        }
    }
    return best;
}

struct FeatureScope {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

struct FingerprintFeatures {
    cv::Mat image;
    cv::Mat gray;
    cv::Mat edges;
    FeatureScope strict;
    FeatureScope expanded;
    FeatureScope anchored;
};

FeatureScope select_feature_scope(
    const FeatureScope& source,
    const cv::Size& image_size,
    double margin_ratio
) {
    FeatureScope result;
    if (source.keypoints.empty()) {
        return result;
    }

    const int margin_y = bankers_round(image_size.height * margin_ratio);
    const int margin_x = bankers_round(image_size.width * margin_ratio);
    const int maximum_y = image_size.height - margin_y;
    const int maximum_x = image_size.width - margin_x;
    std::vector<int> selected_rows;
    selected_rows.reserve(source.keypoints.size());
    result.keypoints.reserve(source.keypoints.size());
    for (int index = 0; index < static_cast<int>(source.keypoints.size()); ++index) {
        const cv::Point2f point = source.keypoints[index].pt;
        const int x = static_cast<int>(point.x + 0.5f);
        const int y = static_cast<int>(point.y + 0.5f);
        if (x >= margin_x && x < maximum_x && y >= margin_y && y < maximum_y) {
            result.keypoints.push_back(source.keypoints[index]);
            selected_rows.push_back(index);
        }
    }

    if (!selected_rows.empty() && !source.descriptors.empty()) {
        result.descriptors.create(
            static_cast<int>(selected_rows.size()),
            source.descriptors.cols,
            source.descriptors.type()
        );
        const std::size_t descriptor_bytes = static_cast<std::size_t>(
            source.descriptors.cols * source.descriptors.elemSize()
        );
        for (int row = 0; row < static_cast<int>(selected_rows.size()); ++row) {
            std::memcpy(
                result.descriptors.ptr(row),
                source.descriptors.ptr(selected_rows[row]),
                descriptor_bytes
            );
        }
    }
    return result;
}

FingerprintFeatures prepare_fingerprint_features(
    const cv::Mat& image,
    const cv::Mat& gray,
    const cv::Mat& edges
) {
    FingerprintFeatures features;
    features.image = image;
    features.gray = gray;
    features.edges = edges;
    thread_local cv::Ptr<cv::SIFT> detector = cv::SIFT::create(
        500, 3, 0.006, 8.0, 0.8
    );
    detector->detectAndCompute(
        features.gray,
        cv::noArray(),
        features.anchored.keypoints,
        features.anchored.descriptors
    );
    features.strict = select_feature_scope(
        features.anchored,
        features.gray.size(),
        kFeatureMarginRatio
    );
    features.expanded = select_feature_scope(
        features.anchored,
        features.gray.size(),
        kExpandedFeatureMarginRatio
    );
    return features;
}

FingerprintFeatures prepare_fingerprint_features(const cv::Mat& image) {
    const cv::Mat gray = to_grayscale(image);
    return prepare_fingerprint_features(
        image,
        gray,
        image_edges_from_gray(gray)
    );
}

bool keypoint_is_central(const cv::KeyPoint& keypoint, const cv::Size& image_size) {
    return keypoint.pt.x >= image_size.width * kFeatureMarginRatio
        && keypoint.pt.x < image_size.width * (1.0 - kFeatureMarginRatio)
        && keypoint.pt.y >= image_size.height * kFeatureMarginRatio
        && keypoint.pt.y < image_size.height * (1.0 - kFeatureMarginRatio);
}

std::vector<cv::DMatch> deduplicate_matches(
    std::vector<cv::DMatch> matches,
    const std::vector<cv::KeyPoint>& keypoints_a,
    const std::vector<cv::KeyPoint>& keypoints_b
) {
    std::stable_sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
        return a.distance < b.distance;
    });
    std::vector<cv::DMatch> accepted;
    std::vector<cv::Point2f> points_a;
    std::vector<cv::Point2f> points_b;
    accepted.reserve(matches.size());
    points_a.reserve(matches.size());
    points_b.reserve(matches.size());
    const double radius_squared = kKeypointDeduplicationRadius * kKeypointDeduplicationRadius;
    for (const cv::DMatch& match : matches) {
        const cv::Point2f point_a = keypoints_a[match.queryIdx].pt;
        const cv::Point2f point_b = keypoints_b[match.trainIdx].pt;
        bool duplicate_a = false;
        bool duplicate_b = false;
        for (const cv::Point2f& existing : points_a) {
            const cv::Point2f difference = point_a - existing;
            duplicate_a = duplicate_a || difference.dot(difference) < radius_squared;
        }
        for (const cv::Point2f& existing : points_b) {
            const cv::Point2f difference = point_b - existing;
            duplicate_b = duplicate_b || difference.dot(difference) < radius_squared;
        }
        if (duplicate_a || duplicate_b) {
            continue;
        }
        accepted.push_back(match);
        points_a.push_back(point_a);
        points_b.push_back(point_b);
    }
    return accepted;
}

int inlier_grid_cell_count(const std::vector<cv::Point2f>& points, const cv::Size& image_size) {
    std::set<std::pair<int, int>> cells;
    for (const cv::Point2f& point : points) {
        const int x = std::min(
            kSpatialVerificationGridSize - 1,
            static_cast<int>(point.x * kSpatialVerificationGridSize / image_size.width)
        );
        const int y = std::min(
            kSpatialVerificationGridSize - 1,
            static_cast<int>(point.y * kSpatialVerificationGridSize / image_size.height)
        );
        cells.emplace(x, y);
    }
    return static_cast<int>(cells.size());
}

struct DirectionalResult {
    double score = 0.0;
    cv::Mat transform;
    int inliers = 0;
    bool complete_spatial = false;
    int central_inliers = 0;
};

int central_inlier_count(
    const std::vector<cv::Point2f>& points_a,
    const std::vector<cv::Point2f>& points_b,
    const cv::Size& image_size
) {
    const double margin_x = image_size.width * kCentralConfirmationMarginRatio;
    const double margin_y = image_size.height * kCentralConfirmationMarginRatio;
    auto is_central = [&](const cv::Point2f& point) {
        return point.x >= margin_x
            && point.x < image_size.width - margin_x
            && point.y >= margin_y
            && point.y < image_size.height - margin_y;
    };
    int count = 0;
    for (std::size_t index = 0; index < points_a.size(); ++index) {
        count += is_central(points_a[index]) && is_central(points_b[index]);
    }
    return count;
}

DirectionalResult directional_fingerprint_score(
    const FeatureScope& scope_a,
    const FeatureScope& scope_b,
    const cv::Size& image_size,
    bool require_central_endpoint,
    bool allow_central_confirmation
) {
    if (scope_a.descriptors.empty() || scope_b.descriptors.rows < 2) {
        return {};
    }
    cv::Mat distances;
    cv::Mat indices;
    cv::batchDistance(
        scope_a.descriptors,
        scope_b.descriptors,
        distances,
        CV_32F,
        indices,
        cv::NORM_L2SQR,
        2
    );
    std::vector<cv::DMatch> matches;
    matches.reserve(distances.rows);
    for (int query = 0; query < distances.rows; ++query) {
        const float* nearest_distances = distances.ptr<float>(query);
        const int* nearest_indices = indices.ptr<int>(query);
        if (nearest_indices[1] < 0
            || nearest_distances[0] >= 0.64f * nearest_distances[1]) {
            continue;
        }
        if (require_central_endpoint
            && !keypoint_is_central(scope_a.keypoints[query], image_size)
            && !keypoint_is_central(scope_b.keypoints[nearest_indices[0]], image_size)) {
            continue;
        }
        matches.emplace_back(query, nearest_indices[0], nearest_distances[0]);
    }
    matches = deduplicate_matches(std::move(matches), scope_a.keypoints, scope_b.keypoints);
    if (matches.size() < 4) {
        return {};
    }

    std::vector<cv::Point2f> points_a;
    std::vector<cv::Point2f> points_b;
    points_a.reserve(matches.size());
    points_b.reserve(matches.size());
    for (const cv::DMatch& match : matches) {
        points_a.push_back(scope_a.keypoints[match.queryIdx].pt);
        points_b.push_back(scope_b.keypoints[match.trainIdx].pt);
    }

    cv::Mat inlier_mask;
    cv::setRNGSeed(0);
    cv::Mat transform = cv::estimateAffinePartial2D(
        points_a,
        points_b,
        inlier_mask,
        cv::RANSAC,
        3.0
    );
    if (transform.empty()) {
        return {};
    }
    const double scale = std::hypot(transform.at<double>(0, 0), transform.at<double>(1, 0));
    if (scale < kMinTransformScale || scale > kMaxTransformScale) {
        return {0.0, transform, 0, false};
    }
    const int inlier_count = inlier_mask.empty() ? 0 : cv::countNonZero(inlier_mask);
    if (inlier_count < kMinConfidentInliers) {
        return {0.0, transform, inlier_count, false};
    }

    std::vector<cv::Point2f> inlier_points_a;
    std::vector<cv::Point2f> inlier_points_b;
    inlier_points_a.reserve(matches.size());
    inlier_points_b.reserve(matches.size());
    for (int index = 0; index < static_cast<int>(matches.size()); ++index) {
        if (inlier_mask.at<unsigned char>(index) != 0) {
            inlier_points_a.push_back(points_a[index]);
            inlier_points_b.push_back(points_b[index]);
        }
    }
    const int central_inliers = central_inlier_count(
        inlier_points_a, inlier_points_b, image_size
    );
    const int required_cells = std::min(kMinSpatialInlierCells, inlier_count);
    const int occupied_cells = std::min(
        inlier_grid_cell_count(inlier_points_a, image_size),
        inlier_grid_cell_count(inlier_points_b, image_size)
    );
    const bool complete_spatial = occupied_cells >= required_cells;
    double spatial_evidence = 1.0;
    if (!complete_spatial) {
        if ((occupied_cells < required_cells - 1
                || inlier_count < kMinFingerprintInliers)
            && (!allow_central_confirmation
                || central_inliers < kMinCentralConfirmationInliers)) {
            return {0.0, transform, inlier_count, false, central_inliers};
        }
        spatial_evidence = kPartialSpatialEvidenceFactor;
    }
    const double geometric_consistency = inlier_count / static_cast<double>(matches.size());
    const double evidence = std::sqrt(std::min(1.0, inlier_count / static_cast<double>(kMinFingerprintInliers)));
    return {
        geometric_consistency * evidence * spatial_evidence,
        transform,
        inlier_count,
        complete_spatial,
        central_inliers,
    };
}

struct ConfidenceMetrics {
    double exponent = 1.0;
    double inverse_error = std::numeric_limits<double>::infinity();
    double maximum_rotation = 0.0;
};

ConfidenceMetrics transform_confidence_metrics(
    const cv::Mat& transform_ab,
    const cv::Mat& transform_ba,
    int inliers_ab,
    int inliers_ba,
    const cv::Size& image_size
) {
    if (transform_ab.empty() || transform_ba.empty()
        || std::min(inliers_ab, inliers_ba) < kMinConfidentInliers) {
        return {};
    }
    const double scale_ab = std::hypot(transform_ab.at<double>(0, 0), transform_ab.at<double>(1, 0));
    const double scale_ba = std::hypot(transform_ba.at<double>(0, 0), transform_ba.at<double>(1, 0));
    if (scale_ab < kMinConfidentScale || scale_ab > kMaxConfidentScale
        || scale_ba < kMinConfidentScale || scale_ba > kMaxConfidentScale) {
        return {};
    }
    constexpr double radians_to_degrees = 180.0 / CV_PI;
    const double rotation_ab = std::abs(std::atan2(
        transform_ab.at<double>(1, 0), transform_ab.at<double>(0, 0)
    ) * radians_to_degrees);
    const double rotation_ba = std::abs(std::atan2(
        transform_ba.at<double>(1, 0), transform_ba.at<double>(0, 0)
    ) * radians_to_degrees);
    const double maximum_rotation = std::max(rotation_ab, rotation_ba);

    auto normalized_affine = [&](const cv::Mat& transform) {
        cv::Matx33d affine(
            transform.at<double>(0, 0), transform.at<double>(0, 1), transform.at<double>(0, 2),
            transform.at<double>(1, 0), transform.at<double>(1, 1), transform.at<double>(1, 2),
            0.0, 0.0, 1.0
        );
        const cv::Matx33d normalize(
            1.0 / image_size.width, 0.0, 0.0,
            0.0, 1.0 / image_size.height, 0.0,
            0.0, 0.0, 1.0
        );
        const cv::Matx33d denormalize(
            image_size.width, 0.0, 0.0,
            0.0, image_size.height, 0.0,
            0.0, 0.0, 1.0
        );
        return normalize * affine * denormalize;
    };
    const cv::Matx33d error_matrix = normalized_affine(transform_ab)
        * normalized_affine(transform_ba) - cv::Matx33d::eye();
    double squared_error = 0.0;
    for (double value : error_matrix.val) {
        squared_error += value * value;
    }
    const double inverse_error = std::sqrt(squared_error);
    if (inverse_error <= kMaxStrongInverseTransformError) {
        return {kStrongFingerprintConfidenceExponent, inverse_error, maximum_rotation};
    }
    if (inverse_error <= kMaxInverseTransformError) {
        return {kFingerprintConfidenceExponent, inverse_error, maximum_rotation};
    }
    const double scale_reciprocity_error = std::abs(std::log(scale_ab * scale_ba));
    if (scale_reciprocity_error <= kMaxScaleReciprocityLogError
        && maximum_rotation >= kMinScaleReciprocalRotation) {
        return {kScaleReciprocalConfidenceExponent, inverse_error, maximum_rotation};
    }
    return {1.0, inverse_error, maximum_rotation};
}

double aligned_grayscale_error(
    const FingerprintFeatures& features_a,
    const FingerprintFeatures& features_b,
    const cv::Mat& transform_ab
) {
    cv::Mat aligned_a;
    cv::Mat valid;
    cv::warpAffine(features_a.gray, aligned_a, transform_ab, features_a.gray.size());
    cv::warpAffine(
        cv::Mat(features_a.gray.size(), CV_8U, cv::Scalar(1)),
        valid,
        transform_ab,
        features_a.gray.size()
    );
    const int margin_y = bankers_round(features_a.gray.rows * kAlignedAppearanceMarginRatio);
    const int margin_x = bankers_round(features_a.gray.cols * kAlignedAppearanceMarginRatio);
    double difference = 0.0;
    std::size_t count = 0;
    for (int y = margin_y; y < features_a.gray.rows - margin_y; ++y) {
        const unsigned char* aligned = aligned_a.ptr<unsigned char>(y);
        const unsigned char* target = features_b.gray.ptr<unsigned char>(y);
        const unsigned char* mask = valid.ptr<unsigned char>(y);
        for (int x = margin_x; x < features_a.gray.cols - margin_x; ++x) {
            if (mask[x] != 0) {
                difference += std::abs(static_cast<int>(aligned[x]) - static_cast<int>(target[x]));
                ++count;
            }
        }
    }
    return count == 0 ? 1.0 : difference / (count * 255.0);
}

double aligned_edge_containment(
    const FingerprintFeatures& features_a,
    const FingerprintFeatures& features_b,
    const cv::Mat& transform_ab
) {
    cv::Mat aligned_a;
    cv::Mat valid;
    cv::warpAffine(features_a.edges, aligned_a, transform_ab, features_a.edges.size());
    cv::warpAffine(
        cv::Mat(features_a.edges.size(), CV_8U, cv::Scalar(1)),
        valid,
        transform_ab,
        features_a.edges.size()
    );
    cv::Mat valid_b;
    cv::bitwise_and(features_b.edges, valid, valid_b);
    cv::Mat nearby_a;
    cv::Mat nearby_b;
    const cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
    cv::dilate(aligned_a, nearby_a, kernel);
    cv::dilate(valid_b, nearby_b, kernel);
    const int count_a = cv::countNonZero(aligned_a);
    const int count_b = cv::countNonZero(valid_b);
    if (count_a == 0 || count_b == 0) {
        return 0.0;
    }
    cv::Mat overlap;
    cv::bitwise_and(aligned_a, nearby_b, overlap);
    const double precision = cv::countNonZero(overlap) / static_cast<double>(count_a);
    cv::bitwise_and(valid_b, nearby_a, overlap);
    const double recall = cv::countNonZero(overlap) / static_cast<double>(count_b);
    return std::max(precision, recall);
}

struct CentralContentMetrics {
    double content_error = 1.0;
    double edge_f1 = 0.0;
    int edge_count_a = 0;
    int edge_count_b = 0;
};

CentralContentMetrics aligned_central_content_metrics(
    const cv::Mat& image_a,
    const cv::Mat& edges_a,
    const cv::Mat& image_b,
    const cv::Mat& edges_b,
    const cv::Mat& transform_ab
) {
    cv::Mat aligned_image_a;
    cv::Mat aligned_edges_a;
    cv::Mat valid;
    cv::warpAffine(image_a, aligned_image_a, transform_ab, image_a.size());
    cv::warpAffine(edges_a, aligned_edges_a, transform_ab, edges_a.size());
    cv::warpAffine(
        cv::Mat(image_a.size(), CV_8U, cv::Scalar(1)),
        valid,
        transform_ab,
        image_a.size()
    );

    const int margin_y = bankers_round(image_a.rows * kCentralContentMarginRatio);
    const int margin_x = bankers_round(image_a.cols * kCentralContentMarginRatio);
    const int channels = image_a.channels();
    double difference = 0.0;
    std::size_t value_count = 0;
    int overlap = 0;
    CentralContentMetrics result;
    for (int y = margin_y; y < image_a.rows - margin_y; ++y) {
        const unsigned char* aligned_pixels = aligned_image_a.ptr<unsigned char>(y);
        const unsigned char* target_pixels = image_b.ptr<unsigned char>(y);
        const unsigned char* aligned_edge = aligned_edges_a.ptr<unsigned char>(y);
        const unsigned char* target_edge = edges_b.ptr<unsigned char>(y);
        const unsigned char* valid_pixel = valid.ptr<unsigned char>(y);
        for (int x = margin_x; x < image_a.cols - margin_x; ++x) {
            if (valid_pixel[x] == 0) {
                continue;
            }
            for (int channel = 0; channel < channels; ++channel) {
                difference += std::abs(
                    static_cast<int>(aligned_pixels[x * channels + channel])
                    - static_cast<int>(target_pixels[x * channels + channel])
                );
                ++value_count;
            }
            const bool has_aligned_edge = aligned_edge[x] != 0;
            const bool has_target_edge = target_edge[x] != 0;
            result.edge_count_a += has_aligned_edge;
            result.edge_count_b += has_target_edge;
            overlap += has_aligned_edge && has_target_edge;
        }
    }
    if (value_count == 0) {
        return result;
    }
    result.content_error = difference / (value_count * 255.0);
    const int edge_total = result.edge_count_a + result.edge_count_b;
    result.edge_f1 = edge_total == 0
        ? 1.0
        : 2.0 * overlap / edge_total;
    return result;
}

double central_content_factor(
    const cv::Mat& image_a,
    const cv::Mat& edges_a,
    const cv::Mat& image_b,
    const cv::Mat& edges_b,
    const cv::Mat& transform_ab,
    const cv::Mat& transform_ba
) {
    const CentralContentMetrics ab = aligned_central_content_metrics(
        image_a, edges_a, image_b, edges_b, transform_ab
    );
    const CentralContentMetrics ba = aligned_central_content_metrics(
        image_b, edges_b, image_a, edges_a, transform_ba
    );
    if (std::min({
            ab.edge_count_a,
            ab.edge_count_b,
            ba.edge_count_a,
            ba.edge_count_b,
        }) < kMinCentralContentEdgePixels) {
        return 1.0;
    }

    const double content_error = std::min(ab.content_error, ba.content_error);
    const double edge_f1 = std::max(ab.edge_f1, ba.edge_f1);
    const double edge_mismatch = clamp01(
        (kCentralContentEdgeWarning - edge_f1)
        / (kCentralContentEdgeWarning - kCentralContentEdgeReject)
    );
    double appearance_mismatch = clamp01(
        (content_error - kCentralContentErrorWarning)
        / (kCentralContentErrorReject - kCentralContentErrorWarning)
    );
    if (edge_f1 >= kCentralContentAppearanceEdgeIgnore) {
        appearance_mismatch = 0.0;
    }
    const double mismatch = std::max(edge_mismatch, appearance_mismatch);
    return 1.0 - kMaxCentralContentPenalty * mismatch;
}

struct ScopeFingerprintResult {
    double score = 0.0;
    std::optional<double> local_content_factor;
};

ScopeFingerprintResult bidirectional_fingerprint_score(
    const FingerprintFeatures& features_a,
    const FeatureScope& scope_a,
    const FingerprintFeatures& features_b,
    const FeatureScope& scope_b,
    bool require_central_endpoint,
    bool allow_central_confirmation,
    bool require_local_content_confirmation,
    const cv::Mat& local_image_a,
    const cv::Mat& local_edges_a,
    const cv::Mat& local_image_b,
    const cv::Mat& local_edges_b
) {
    const cv::Size image_size = features_a.gray.size();
    const DirectionalResult ab = directional_fingerprint_score(
        scope_a,
        scope_b,
        image_size,
        require_central_endpoint,
        allow_central_confirmation
    );
    if (ab.score == 0.0) {
        return {};
    }
    const DirectionalResult ba = directional_fingerprint_score(
        scope_b,
        scope_a,
        image_size,
        require_central_endpoint,
        allow_central_confirmation
    );
    if (ba.score == 0.0) {
        return {};
    }
    double score = 2.0 * ab.score * ba.score / (ab.score + ba.score);
    const double base_score = score;
    const ConfidenceMetrics confidence = transform_confidence_metrics(
        ab.transform, ba.transform, ab.inliers, ba.inliers, image_size
    );
    if (!(ab.complete_spatial && ba.complete_spatial) && confidence.exponent <= 1.0) {
        return {};
    }
    if (confidence.exponent > 1.0) {
        score = 1.0 - std::pow(1.0 - score, confidence.exponent);
        const double aligned_error = std::min(
            aligned_grayscale_error(features_a, features_b, ab.transform),
            aligned_grayscale_error(features_b, features_a, ba.transform)
        );
        const double edge_containment = std::max(
            aligned_edge_containment(features_a, features_b, ab.transform),
            aligned_edge_containment(features_b, features_a, ba.transform)
        );
        const double background_strength = clamp01(
            (edge_containment - kMinBackgroundEdgeContainment)
            / (1.0 - kMinBackgroundEdgeContainment)
        );
        const double inlier_support = std::min(ab.inliers, ba.inliers)
            / static_cast<double>(kMinFingerprintInliers);
        double strong_support = 0.0;
        if (base_score >= kMinStrongGeometricScore) {
            strong_support = clamp01(
                (inlier_support - kMinStrongInlierSupport)
                / (kFullStrongInlierSupport - kMinStrongInlierSupport)
            );
        }
        const double background_allowance = std::max(
            background_strength * kMaxBackgroundErrorAllowance,
            strong_support * kMaxStrongInlierErrorAllowance
        );
        const double allowed_error = kMaxAlignedAppearanceError + background_allowance;
        if (aligned_error > allowed_error) {
            score *= std::exp(
                -(aligned_error - allowed_error) / kAlignedAppearanceErrorDecay
            );
        }
    } else if (std::isfinite(confidence.inverse_error)
        && confidence.inverse_error > kMaxInverseTransformError
        && confidence.maximum_rotation < kMinScaleReciprocalRotation) {
        score *= std::pow(
            appearance_score(features_a.image, features_b.image),
            kInconsistentAppearanceExponent
        );
    }
    const int central_support = allow_central_confirmation
        ? std::min(ab.central_inliers, ba.central_inliers)
        : 0;
    const double central_strength = clamp01(
        (central_support - kMinCentralConfirmationInliers + 1.0)
        / (kFullCentralConfirmationInliers - kMinCentralConfirmationInliers + 1.0)
    );
    const double central_exponent = 1.0 + central_strength
        * (kCentralConfirmationConfidenceExponent - 1.0);
    score = 1.0 - std::pow(1.0 - score, central_exponent);
    ScopeFingerprintResult result;
    result.score = score;
    if (require_local_content_confirmation
        && ab.complete_spatial
        && ba.complete_spatial) {
        result.local_content_factor = central_content_factor(
            local_image_a,
            local_edges_a,
            local_image_b,
            local_edges_b,
            ab.transform,
            ba.transform
        );
    }
    return result;
}

double fingerprint_score(
    const FingerprintFeatures& a,
    const FingerprintFeatures& b,
    bool allow_central_confirmation,
    bool require_local_content_confirmation,
    const cv::Mat& local_image_a,
    const cv::Mat& local_edges_a,
    const cv::Mat& local_image_b,
    const cv::Mat& local_edges_b
) {
    const ScopeFingerprintResult strict = bidirectional_fingerprint_score(
        a,
        a.strict,
        b,
        b.strict,
        false,
        allow_central_confirmation,
        require_local_content_confirmation,
        local_image_a,
        local_edges_a,
        local_image_b,
        local_edges_b
    );
    if (strict.score == 1.0
        && strict.local_content_factor.value_or(1.0) == 1.0) {
        return 1.0;
    }
    const ScopeFingerprintResult expanded = bidirectional_fingerprint_score(
        a,
        a.expanded,
        b,
        b.expanded,
        false,
        allow_central_confirmation,
        require_local_content_confirmation,
        local_image_a,
        local_edges_a,
        local_image_b,
        local_edges_b
    );
    const ScopeFingerprintResult anchored = bidirectional_fingerprint_score(
        a,
        a.anchored,
        b,
        b.anchored,
        true,
        allow_central_confirmation,
        require_local_content_confirmation,
        local_image_a,
        local_edges_a,
        local_image_b,
        local_edges_b
    );
    const double secondary = std::max(expanded.score, anchored.score);
    double score = 0.0;
    if (strict.score == 0.0) {
        score = secondary * kMissingStrictScopeFactor;
    } else {
        score = strict.score * kStrictScopeWeight
            + std::max(strict.score, secondary) * (1.0 - kStrictScopeWeight);
    }

    // A wider SIFT scope must not bypass negative evidence from a verified
    // strict-scope alignment.
    std::optional<double> local_factor = strict.local_content_factor;
    if (!local_factor.has_value()) {
        if (expanded.local_content_factor.has_value()) {
            local_factor = expanded.local_content_factor;
        }
        if (anchored.local_content_factor.has_value()) {
            local_factor = local_factor.has_value()
                ? std::min(*local_factor, *anchored.local_content_factor)
                : anchored.local_content_factor;
        }
    }
    return score * local_factor.value_or(1.0);
}

struct PreparedTile {
    explicit PreparedTile(const cv::Mat& source, bool prepare_raw_features)
        : image(source.clone()),
          digest(content_digest(image)),
          gray(to_grayscale(image)),
          edges(image_edges_from_gray(gray)),
          edge_bits(edge_rows(edges)),
          edge_total(edge_count(edge_bits)),
          gradient_features(prepare_fingerprint_features(gradient_magnitude(gray))) {
        if (prepare_raw_features) {
            raw_features.emplace(
                prepare_fingerprint_features(image, gray, edges)
            );
        }
    }

    FingerprintFeatures& ensure_raw_features() const {
        if (!raw_features.has_value()) {
            raw_features.emplace(
                prepare_fingerprint_features(image, gray, edges)
            );
        }
        return *raw_features;
    }

    cv::Mat image;
    std::array<unsigned char, 16> digest;
    cv::Mat gray;
    cv::Mat edges;
    std::array<std::uint64_t, kWorkingSize> edge_bits;
    unsigned int edge_total;
    FingerprintFeatures gradient_features;
    mutable std::optional<FingerprintFeatures> raw_features;
};

struct PreparedImageData {
    std::vector<PreparedTile> tiles;
};

std::unique_ptr<PreparedImageData> prepare_image(
    const cv::Mat& image,
    bool prepare_raw_features
) {
    const cv::Size target = normalized_size(image);
    cv::Mat normalized;
    if (image.size() == target) {
        normalized = image;
    } else {
        cv::resize(image, normalized, target, 0.0, 0.0, cv::INTER_CUBIC);
    }

    auto result = std::make_unique<PreparedImageData>();
    const std::vector<cv::Mat> tiles = split_working_tiles(normalized);
    result->tiles.reserve(tiles.size());
    const bool prepare_tile_raw_features = prepare_raw_features && tiles.size() == 1;
    for (const cv::Mat& tile : tiles) {
        result->tiles.emplace_back(tile, prepare_tile_raw_features);
    }
    return result;
}

double picture_match_prepared_64(
    const PreparedTile& tile_a,
    const PreparedTile& tile_b,
    bool require_grayscale_confirmation
) {
    const PreparedTile* ordered_a = &tile_a;
    const PreparedTile* ordered_b = &tile_b;
    if (tile_b.digest < tile_a.digest) {
        ordered_a = &tile_b;
        ordered_b = &tile_a;
    }
    double outline_score = (
        best_shifted_f1(
            ordered_a->edge_bits,
            ordered_a->edge_total,
            ordered_b->edge_bits
        )
        + best_shifted_f1(
            ordered_b->edge_bits,
            ordered_b->edge_total,
            ordered_a->edge_bits
        )
    ) / 2.0;
    outline_score *= appearance_score(ordered_a->image, ordered_b->image);
    if (outline_score == 1.0) {
        return 1.0;
    }

    double fingerprint = fingerprint_score(
        ordered_a->gradient_features,
        ordered_b->gradient_features,
        true,
        require_grayscale_confirmation,
        ordered_a->image,
        ordered_a->edges,
        ordered_b->image,
        ordered_b->edges
    );
    // Raw confirmation only lowers fingerprint, so it cannot beat outline here.
    if (require_grayscale_confirmation
        && fingerprint > 0.0
        && fingerprint < kMinSelfSufficientFingerprintScore
        && fingerprint > outline_score) {
        fingerprint *= std::sqrt(fingerprint_score(
            ordered_a->ensure_raw_features(),
            ordered_b->ensure_raw_features(),
            false,
            false,
            ordered_a->image,
            ordered_a->edges,
            ordered_b->image,
            ordered_b->edges
        ));
    }
    return std::max(outline_score, fingerprint);
}

double match_prepared_images(
    const PreparedImageData& a,
    const PreparedImageData& b
) {
    if (a.tiles.size() != b.tiles.size()) {
        throw std::runtime_error("normalized image tile counts differ");
    }
    const bool require_grayscale_confirmation = a.tiles.size() == 1;
    double total = 0.0;
    double squared_total = 0.0;
    for (std::size_t index = 0; index < a.tiles.size(); ++index) {
        const double score = picture_match_prepared_64(
            a.tiles[index],
            b.tiles[index],
            require_grayscale_confirmation
        );
        total += score;
        squared_total += score * score;
    }
    const double tile_count = static_cast<double>(a.tiles.size());
    const double arithmetic_mean = total / tile_count;
    if (a.tiles.size() < 4 || total == 0.0 || squared_total == 0.0) {
        return arithmetic_mean;
    }

    // Discount isolated failed tiles only when useful evidence is distributed
    // across most of a long image. One accidental high tile therefore keeps
    // the ordinary mean, while repeated corresponding structures can agree.
    const double effective_tile_count = total * total / squared_total;
    const double blend = clamp01(
        (effective_tile_count - tile_count / 2.0) / (tile_count / 6.0)
    );
    const double evidence_weighted_mean = squared_total / total;
    return arithmetic_mean
        + blend * (evidence_weighted_mean - arithmetic_mean);
}

class PreparedImage {
public:
    explicit PreparedImage(const cv::Mat& image)
        : rows_(image.rows),
          cols_(image.cols),
          channels_(image.channels()),
          prepared_(prepare_image(image, true)) {
    }

    int rows() const {
        return rows_;
    }

    int cols() const {
        return cols_;
    }

    int channels() const {
        return channels_;
    }

    const PreparedImageData& prepared() const {
        return *prepared_;
    }

private:
    int rows_;
    int cols_;
    int channels_;
    std::unique_ptr<PreparedImageData> prepared_;
};

void validate_compatible_images(const cv::Mat& image_a, const cv::Mat& image_b) {
    if (image_a.rows != image_b.rows
        || image_a.cols != image_b.cols
        || image_a.channels() != image_b.channels()) {
        throw std::invalid_argument(
            "images must have identical width, height, and channel layout"
        );
    }
}

void validate_compatible_images(
    const PreparedImage& image_a,
    const PreparedImage& image_b
) {
    if (image_a.rows() != image_b.rows()
        || image_a.cols() != image_b.cols()
        || image_a.channels() != image_b.channels()) {
        throw std::invalid_argument(
            "images must have identical width, height, and channel layout"
        );
    }
}

double picture_match_impl(const cv::Mat& image_a, const cv::Mat& image_b) {
    validate_compatible_images(image_a, image_b);
    std::unique_ptr<PreparedImageData> a = prepare_image(image_a, false);
    std::unique_ptr<PreparedImageData> b = prepare_image(image_b, false);
    return match_prepared_images(*a, *b);
}

double picture_match_prepared(
    const PreparedImage& image_a,
    const PreparedImage& image_b
) {
    validate_compatible_images(image_a, image_b);
    return match_prepared_images(image_a.prepared(), image_b.prepared());
}

double picture_match_prepared_cold(
    const PreparedImage& image_a,
    const cv::Mat& image_b
) {
    if (image_a.rows() != image_b.rows
        || image_a.cols() != image_b.cols
        || image_a.channels() != image_b.channels()) {
        throw std::invalid_argument(
            "images must have identical width, height, and channel layout"
        );
    }
    std::unique_ptr<PreparedImageData> b = prepare_image(image_b, false);
    return match_prepared_images(image_a.prepared(), *b);
}

double picture_match_64_impl(const cv::Mat& image_a, const cv::Mat& image_b) {
    validate_compatible_images(image_a, image_b);
    if (image_a.rows != kWorkingSize || image_a.cols != kWorkingSize
        || image_b.rows != kWorkingSize || image_b.cols != kWorkingSize) {
        throw std::invalid_argument("picture_match_64 requires two 64 x 64 images");
    }
    PreparedTile a(image_a, false);
    PreparedTile b(image_b, false);
    return picture_match_prepared_64(a, b, true);
}

}  // namespace

PYBIND11_MODULE(_native, module) {
    cv::setNumThreads(1);
    if (cv::getNumThreads() != 1) {
        throw std::runtime_error("OpenCV single-thread mode could not be enabled");
    }

    module.doc() = "OpenCV-backed native image matching core.";
    py::class_<PreparedImage>(module, "PreparedImage")
        .def(py::init([](const py::array& image) {
            cv::Mat owned = image_from_array(image, "image");
            py::gil_scoped_release release;
            return std::make_unique<PreparedImage>(std::move(owned));
        }))
        .def("match", [](const PreparedImage& self, const PreparedImage& other) {
            py::gil_scoped_release release;
            return picture_match_prepared(self, other);
        })
        .def("match", [](const PreparedImage& self, const py::array& other) {
            cv::Mat owned = image_from_array(other, "image");
            py::gil_scoped_release release;
            return picture_match_prepared_cold(self, owned);
        });

    module.def("picture_match", [](const py::array& image_a, const py::array& image_b) {
        cv::Mat a = image_from_array(image_a, "image_a");
        cv::Mat b = image_from_array(image_b, "image_b");
        py::gil_scoped_release release;
        return picture_match_impl(a, b);
    });
    module.def("picture_match_64", [](const py::array& image_a, const py::array& image_b) {
        cv::Mat a = image_from_array(image_a, "image_a");
        cv::Mat b = image_from_array(image_b, "image_b");
        py::gil_scoped_release release;
        return picture_match_64_impl(a, b);
    });
    module.def(
        "resize_image",
        [](const py::array& image, int target_height, int target_width) {
            if (target_height <= 0 || target_width <= 0) {
                throw std::invalid_argument("target dimensions must be positive");
            }
            const bool preserve_channel_axis = image.ndim() == 3;
            cv::Mat source = image_from_array(image, "image");
            cv::Mat resized;
            {
                py::gil_scoped_release release;
                cv::resize(
                    source,
                    resized,
                    {target_width, target_height},
                    0.0,
                    0.0,
                    cv::INTER_CUBIC
                );
            }
            return image_to_array(resized, preserve_channel_axis);
        },
        py::arg("image"),
        py::arg("target_height"),
        py::arg("target_width")
    );
    module.def("clear_feature_cache", []() {});
}
