/**
 * 网页端 K-means 图片聚类。
 * 浏览器负责解码 PNG/JPEG/WebP/GIF/BMP/AVIF/SVG 等可支持格式。
 */
async function kmeansImageClusterFromFile(file, k, options = {}) {
    if (!Number.isInteger(k) || k <= 0) throw new Error('K值必须是一个大于0的整数');

    const colorSpace = options.colorSpace || 'RGB';
    if (!['RGB', 'LAB'].includes(colorSpace)) {
        throw new Error("不支持的色彩空间，仅支持 'RGB' 或 'LAB'");
    }

    const maxDimension = options.maxDimension || 150;
    const maxIterations = options.maxIterations || 20;
    const convergenceThreshold = options.convergenceThreshold || 0.001;
    const decoded = await decodeImageFile(file);
    const source = decoded.source;

    let width = source.width || source.naturalWidth;
    let height = source.height || source.naturalHeight;
    if (!width || !height) throw new Error('无法读取图片尺寸');

    if (width > maxDimension || height > maxDimension) {
        const ratio = Math.min(maxDimension / width, maxDimension / height);
        width = Math.max(1, Math.round(width * ratio));
        height = Math.max(1, Math.round(height * ratio));
    }

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(source, 0, 0, width, height);

    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;
    const pixelCount = width * height;
    if (k > pixelCount) throw new Error('K值不能大于处理后的像素数量');

    const features = new Float64Array(pixelCount * 3);
    for (let i = 0; i < pixelCount; i++) {
        const rgbaIdx = i * 4;
        const rgbIdx = i * 3;
        const alpha = data[rgbaIdx + 3] / 255;
        const r = Math.round(data[rgbaIdx] * alpha + 255 * (1 - alpha));
        const g = Math.round(data[rgbaIdx + 1] * alpha + 255 * (1 - alpha));
        const b = Math.round(data[rgbaIdx + 2] * alpha + 255 * (1 - alpha));

        if (colorSpace === 'LAB') {
            const lab = rgb2lab(r, g, b);
            features[rgbIdx] = lab[0];
            features[rgbIdx + 1] = lab[1];
            features[rgbIdx + 2] = lab[2];
        } else {
            features[rgbIdx] = r;
            features[rgbIdx + 1] = g;
            features[rgbIdx + 2] = b;
        }
    }

    const centroids = initializeCentroids(features, pixelCount, k, colorSpace);
    const labels = new Int32Array(pixelCount);
    let iterations = 0;
    let converged = false;

    while (!converged && iterations < maxIterations) {
        iterations++;
        let maxShift = 0;

        for (let p = 0; p < pixelCount; p++) {
            const idx = p * 3;
            let minDist = Infinity;
            let bestCluster = 0;
            for (let c = 0; c < k; c++) {
                const dist = calculateDistanceAt(features, idx, centroids[c], colorSpace);
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = c;
                }
            }
            labels[p] = bestCluster;
        }

        const sums = Array.from({ length: k }, () => [0, 0, 0]);
        const counts = new Array(k).fill(0);

        for (let p = 0; p < pixelCount; p++) {
            const idx = p * 3;
            const c = labels[p];
            sums[c][0] += features[idx];
            sums[c][1] += features[idx + 1];
            sums[c][2] += features[idx + 2];
            counts[c]++;
        }

        for (let c = 0; c < k; c++) {
            if (counts[c] === 0) {
                const reseeded = reseedEmptyCluster(c, labels, features, centroids, pixelCount, colorSpace);
                sums[c][0] = reseeded[0];
                sums[c][1] = reseeded[1];
                sums[c][2] = reseeded[2];
                counts[c] = 1;
            }
        }

        for (let c = 0; c < k; c++) {
            if (counts[c] > 0) {
                const newCentroid = [
                    sums[c][0] / counts[c],
                    sums[c][1] / counts[c],
                    sums[c][2] / counts[c]
                ];
                const shift = calculateDistance(newCentroid, centroids[c], colorSpace);
                if (shift > maxShift) maxShift = shift;
                centroids[c] = newCentroid;
            }
        }

        if (maxShift < convergenceThreshold) converged = true;
    }

    const clusteredData = new Uint8ClampedArray(data.length);
    for (let p = 0; p < pixelCount; p++) {
        const rgbaIdx = p * 4;
        const c = labels[p];
        const rgb = centroidToRgb(centroids[c], colorSpace);
        clusteredData[rgbaIdx] = rgb[0];
        clusteredData[rgbaIdx + 1] = rgb[1];
        clusteredData[rgbaIdx + 2] = rgb[2];
        clusteredData[rgbaIdx + 3] = 255;
    }

    const clusteredCanvas = document.createElement('canvas');
    clusteredCanvas.width = width;
    clusteredCanvas.height = height;
    clusteredCanvas.getContext('2d').putImageData(new ImageData(clusteredData, width, height), 0, 0);
    const clusteredImageBase64 = clusteredCanvas.toDataURL('image/png');

    const counts = new Array(k).fill(0);
    for (let p = 0; p < pixelCount; p++) counts[labels[p]]++;

    const palette = centroids.map((centroid, i) => {
        const [r, g, b] = centroidToRgb(centroid, colorSpace);
        const hex = '#' + [r, g, b].map(c => c.toString(16).padStart(2, '0')).join('');
        return {
            r, g, b,
            hex,
            ratio: counts[i] / pixelCount,
            pixelCount: counts[i],
            percentage: (counts[i] / pixelCount * 100).toFixed(2) + '%'
        };
    }).sort((a, b) => b.ratio - a.ratio);

    const chartData = buildChartData(palette, pixelCount, k);
    return {
        meta: {
            processedWidth: width,
            processedHeight: height,
            totalPixels: pixelCount,
            iterations,
            converged,
            k,
            colorSpace,
            maxDimension,
            convergenceThreshold,
            sourceType: file.type || 'unknown',
            sourceName: file.name
        },
        palette,
        chartData,
        echartsOptions: buildEchartsOptions(chartData),
        clusteredImageBase64,
        originalImageUrl: decoded.url
    };
}

async function decodeImageFile(file) {
    const looksLikeImage = file.type.startsWith('image/') ||
        /\.(avif|bmp|gif|jpe?g|png|svg|webp)$/i.test(file.name);
    if (!looksLikeImage) {
        throw new Error('请选择图片文件，支持浏览器可解码的 PNG、JPG、WebP、GIF、BMP、AVIF、SVG 等格式');
    }

    const url = URL.createObjectURL(file);
    if ('createImageBitmap' in window) {
        try {
            const bitmap = await createImageBitmap(file);
            return { source: bitmap, url };
        } catch (err) {
            // 部分浏览器不支持用 createImageBitmap 解码 SVG，继续使用 Image 回退。
        }
    }

    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve({ source: img, url });
        img.onerror = () => {
            URL.revokeObjectURL(url);
            reject(new Error('当前浏览器无法解码该图片格式，请换用 PNG、JPG、WebP、GIF、BMP、AVIF 或 SVG'));
        };
        img.src = url;
    });
}

function initializeCentroids(features, pixelCount, k, colorSpace) {
    const centroids = [];
    const firstIdx = Math.floor(Math.random() * pixelCount) * 3;
    centroids.push([features[firstIdx], features[firstIdx + 1], features[firstIdx + 2]]);

    for (let i = 1; i < k; i++) {
        const distances = new Float32Array(pixelCount);
        let totalDist = 0;
        for (let p = 0; p < pixelCount; p++) {
            const idx = p * 3;
            let minDist = Infinity;
            for (let c = 0; c < centroids.length; c++) {
                const dist = calculateDistanceAt(features, idx, centroids[c], colorSpace);
                if (dist < minDist) minDist = dist;
            }
            distances[p] = minDist;
            totalDist += minDist;
        }

        if (totalDist === 0) {
            const idx = Math.floor(Math.random() * pixelCount) * 3;
            centroids.push([features[idx], features[idx + 1], features[idx + 2]]);
            continue;
        }

        let target = Math.random() * totalDist;
        let selected = pixelCount - 1;
        for (let p = 0; p < pixelCount; p++) {
            target -= distances[p];
            if (target <= 0) {
                selected = p;
                break;
            }
        }
        const idx = selected * 3;
        centroids.push([features[idx], features[idx + 1], features[idx + 2]]);
    }

    return centroids;
}

function reseedEmptyCluster(clusterIndex, labels, features, centroids, pixelCount, colorSpace) {
    let maxCount = 0;
    let maxCluster = 0;
    const counts = new Array(centroids.length).fill(0);
    for (let p = 0; p < pixelCount; p++) counts[labels[p]]++;

    for (let j = 0; j < counts.length; j++) {
        if (counts[j] > maxCount) {
            maxCount = counts[j];
            maxCluster = j;
        }
    }

    let maxDist = -1;
    let newCenterPixel = 0;
    for (let p = 0; p < pixelCount; p++) {
        if (labels[p] === maxCluster) {
            const idx = p * 3;
            const dist = calculateDistanceAt(features, idx, centroids[maxCluster], colorSpace);
            if (dist > maxDist) {
                maxDist = dist;
                newCenterPixel = p;
            }
        }
    }

    const idx = newCenterPixel * 3;
    centroids[clusterIndex] = [features[idx], features[idx + 1], features[idx + 2]];
    return centroids[clusterIndex];
}

function calculateDistanceAt(features, idx, centroid, colorSpace) {
    const d0 = features[idx] - centroid[0];
    const d1 = features[idx + 1] - centroid[1];
    const d2 = features[idx + 2] - centroid[2];
    return colorSpace === 'LAB' ? d0 * d0 * 1.2 + d1 * d1 + d2 * d2 : d0 * d0 + d1 * d1 + d2 * d2;
}

function calculateDistance(vec1, vec2, colorSpace) {
    const d0 = vec1[0] - vec2[0];
    const d1 = vec1[1] - vec2[1];
    const d2 = vec1[2] - vec2[2];
    return colorSpace === 'LAB' ? d0 * d0 * 1.2 + d1 * d1 + d2 * d2 : d0 * d0 + d1 * d1 + d2 * d2;
}

function centroidToRgb(centroid, colorSpace) {
    let r, g, b;
    if (colorSpace === 'LAB') {
        [r, g, b] = lab2rgb(centroid[0], centroid[1], centroid[2]);
    } else {
        r = Math.round(centroid[0]);
        g = Math.round(centroid[1]);
        b = Math.round(centroid[2]);
    }
    return [
        Math.max(0, Math.min(255, r)),
        Math.max(0, Math.min(255, g)),
        Math.max(0, Math.min(255, b))
    ];
}

function buildChartData(palette, totalPixels, k) {
    return {
        pieData: palette.map((p, index) => ({
            name: '类簇 ' + (index + 1),
            value: p.pixelCount,
            itemStyle: { color: p.hex },
            customData: {
                hex: p.hex,
                rgb: `(${p.r}, ${p.g}, ${p.b})`,
                ratio: p.ratio,
                percentage: p.percentage
            }
        })),
        barData: palette.map((p, index) => ({
            name: '类簇 ' + (index + 1),
            value: p.pixelCount,
            itemStyle: { color: p.hex },
            customData: {
                hex: p.hex,
                rgb: `(${p.r}, ${p.g}, ${p.b})`,
                ratio: p.ratio,
                percentage: p.percentage
            }
        })),
        colors: palette.map(p => p.hex),
        values: palette.map(p => p.pixelCount),
        ratios: palette.map(p => p.ratio),
        colorNames: palette.map(p => `${p.hex} (${p.r},${p.g},${p.b})`),
        treemapData: palette.map((p, index) => ({
            name: '类簇 ' + (index + 1),
            value: p.pixelCount,
            itemStyle: { color: p.hex },
            customData: {
                hex: p.hex,
                rgb: `(${p.r}, ${p.g}, ${p.b})`,
                percentage: p.percentage
            }
        })),
        stats: {
            totalPixels,
            totalClusters: k,
            maxCluster: palette[0]?.pixelCount || 0,
            minCluster: palette[palette.length - 1]?.pixelCount || 0
        }
    };
}

function buildEchartsOptions(chartData) {
    return {
        pieOption: {
            title: { text: '颜色占比分布', left: 'center', textStyle: { fontSize: 14 } },
            tooltip: {
                trigger: 'item',
                formatter: function(params) {
                    const data = params.data.customData || {};
                    return '<strong>' + params.name + '</strong><br/>' +
                        '颜色: ' + (data.hex || 'N/A') + '<br/>' +
                        'RGB: ' + (data.rgb || 'N/A') + '<br/>' +
                        '像素数量: ' + params.value.toLocaleString() + '<br/>' +
                        '占比: ' + (data.percentage || 'N/A');
                }
            },
            legend: { orient: 'vertical', left: 'left' },
            series: [{
                name: '颜色分布',
                type: 'pie',
                radius: ['40%', '70%'],
                avoidLabelOverlap: true,
                itemStyle: { borderRadius: 8, borderColor: '#fff', borderWidth: 2 },
                label: {
                    show: true,
                    formatter: function(params) {
                        const data = params.data.customData || {};
                        return params.name + '\n' + (data.hex || '') + '\n' + params.value.toLocaleString() + 'px';
                    }
                },
                data: chartData.pieData
            }]
        },
        barOption: {
            title: { text: '颜色像素数量', left: 'center', textStyle: { fontSize: 14 } },
            tooltip: {
                trigger: 'axis',
                formatter: function(params) {
                    const data = params[0].data.customData || {};
                    return '<strong>' + params[0].name + '</strong><br/>' +
                        '颜色: ' + (data.hex || 'N/A') + '<br/>' +
                        'RGB: ' + (data.rgb || 'N/A') + '<br/>' +
                        '像素数量: ' + params[0].value.toLocaleString() + '<br/>' +
                        '占比: ' + (data.percentage || 'N/A');
                }
            },
            grid: { left: '3%', right: '4%', bottom: '3%', top: '15%', containLabel: true },
            xAxis: {
                type: 'category',
                data: chartData.colorNames,
                axisLabel: { rotate: 0, fontSize: 10, interval: 0 }
            },
            yAxis: {
                type: 'value',
                name: '像素数量',
                axisLabel: {
                    formatter: function(value) {
                        return value >= 1000 ? (value / 1000).toFixed(1) + 'k' : value.toString();
                    }
                }
            },
            series: [{
                name: '颜色分布',
                type: 'bar',
                data: chartData.barData,
                barWidth: '60%',
                label: {
                    show: true,
                    position: 'top',
                    formatter: function(params) {
                        return params.value.toLocaleString();
                    }
                }
            }]
        },
        treemapOption: {
            title: { text: '颜色面积占比', left: 'center', textStyle: { fontSize: 14 } },
            tooltip: {
                formatter: function(info) {
                    const data = info.data.customData || {};
                    return '<strong>' + info.name + '</strong><br/>' +
                        '颜色: ' + (data.hex || 'N/A') + '<br/>' +
                        'RGB: ' + (data.rgb || 'N/A') + '<br/>' +
                        '像素数量: ' + info.value.toLocaleString() + '<br/>' +
                        '占比: ' + (data.percentage || 'N/A');
                }
            },
            series: [{
                type: 'treemap',
                top: 48,
                bottom: 8,
                left: 8,
                right: 8,
                roam: false,
                nodeClick: false,
                breadcrumb: { show: false },
                label: {
                    show: true,
                    formatter: function(params) {
                        const data = params.data.customData || {};
                        return params.name + '\n' + (data.hex || '') + '\n' + (data.percentage || '');
                    },
                    color: '#ffffff',
                    textBorderColor: 'rgba(0,0,0,0.35)',
                    textBorderWidth: 2
                },
                itemStyle: {
                    borderColor: '#ffffff',
                    borderWidth: 3,
                    gapWidth: 3
                },
                data: chartData.treemapData
            }]
        }
    };
}

function rgb2lab(r, g, b) {
    r = r / 255; g = g / 255; b = b / 255;

    r = r > 0.04045 ? Math.pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
    g = g > 0.04045 ? Math.pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
    b = b > 0.04045 ? Math.pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

    let x = r * 0.4124 + g * 0.3576 + b * 0.1805;
    let y = r * 0.2126 + g * 0.7152 + b * 0.0722;
    let z = r * 0.0193 + g * 0.1192 + b * 0.9505;

    x = x / 0.95047; y = y / 1.00000; z = z / 1.08883;

    x = x > 0.008856 ? Math.pow(x, 1 / 3) : (7.787 * x) + (16 / 116);
    y = y > 0.008856 ? Math.pow(y, 1 / 3) : (7.787 * y) + (16 / 116);
    z = z > 0.008856 ? Math.pow(z, 1 / 3) : (7.787 * z) + (16 / 116);

    const l = (116 * y) - 16;
    const a = 500 * (x - y);
    const bVal = 200 * (y - z);

    return [l, a, bVal];
}

function lab2rgb(l, a, b) {
    let y = (l + 16) / 116;
    let x = a / 500 + y;
    let z = y - b / 200;

    x = Math.pow(x, 3) > 0.008856 ? Math.pow(x, 3) : (x - 16 / 116) / 7.787;
    y = Math.pow(y, 3) > 0.008856 ? Math.pow(y, 3) : (y - 16 / 116) / 7.787;
    z = Math.pow(z, 3) > 0.008856 ? Math.pow(z, 3) : (z - 16 / 116) / 7.787;

    x = x * 0.95047; y = y * 1.00000; z = z * 1.08883;

    let r = x * 3.2406 + y * -1.5372 + z * -0.4986;
    let g = x * -0.9689 + y * 1.8758 + z * 0.0415;
    let bVal = x * 0.0557 + y * -0.2040 + z * 1.0570;

    r = r > 0.0031308 ? 1.055 * Math.pow(r, 1 / 2.4) - 0.055 : 12.92 * r;
    g = g > 0.0031308 ? 1.055 * Math.pow(g, 1 / 2.4) - 0.055 : 12.92 * g;
    bVal = bVal > 0.0031308 ? 1.055 * Math.pow(bVal, 1 / 2.4) - 0.055 : 12.92 * bVal;

    r = Math.max(0, Math.min(255, Math.round(r * 255)));
    g = Math.max(0, Math.min(255, Math.round(g * 255)));
    bVal = Math.max(0, Math.min(255, Math.round(bVal * 255)));

    return [r, g, bVal];
}
