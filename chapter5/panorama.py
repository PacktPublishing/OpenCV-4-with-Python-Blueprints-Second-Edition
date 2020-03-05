import argparse
from pathlib import Path
import numpy as np
from hdr import load_image
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    img_group = parser.add_mutually_exclusive_group(required=True)
    img_group.add_argument('--image-dir', type=Path)
    img_group.add_argument('--images', type=Path, nargs='+')
    parser.add_argument('--show-steps', action='store_true')
    args = parser.parse_args()

    if args.image_dir:
        args.images = sorted(args.image_dir.iterdir())
    return args


def largest_connected_subset(images):
    finder = cv2.xfeatures2d_SURF.create()
    all_img_features = [cv2.detail.computeImageFeatures2(finder, img)
                        for img in images]

    matcher = cv2.detail.BestOf2NearestMatcher_create(False, 0.6)
    pair_matches = matcher.apply2(all_img_features)
    matcher.collectGarbage()

    _conn_indices = cv2.detail.leaveBiggestComponent(all_img_features, pair_matches, 0.4)
    conn_indices = [i for [i] in _conn_indices]
    if len(conn_indices) < 2:
        raise RuntimeError("Need 2 or more connected images.")

    conn_features = np.array([all_img_features[i] for i in conn_indices])
    conn_images = [images[i] for i in conn_indices]

    if len(conn_images) < len(images):
        pair_matches = matcher.apply2(conn_features)
        matcher.collectGarbage()

    return conn_images, conn_features, pair_matches


def find_camera_parameters(features, pair_matches):
    estimator = cv2.detail_HomographyBasedEstimator()
    success, cameras = estimator.apply(features, pair_matches, None)
    if not success:
        raise RuntimeError("Homography estimation failed.")

    for cam in cameras:
        cam.R = cam.R.astype(np.float32)

    adjuster = cv2.detail_BundleAdjusterRay()
    adjuster.setConfThresh(0.8)

    refine_mask = np.array([[1, 1, 1],
                            [0, 1, 1],
                            [0, 0, 0]], dtype=np.uint8)
    adjuster.setRefinementMask(refine_mask)

    success, cameras = adjuster.apply(features, p, cameras)

    if not success:
        raise RuntimeError("Camera parameters adjusting failed.")

    print(cameras)
    return cameras


if __name__ == '__main__':
    args = parse_args()
    all_images = [load_image(p, bps=8) for p in args.images]


    conn_images, features, p = largest_connected_subset(all_images)

    cameras = find_camera_parameters(features, p)

    focals = [cam.focal for cam in cameras]
    warped_image_scale = np.mean(focals)

    # corners, sizes, images_warped, masks_warped = [], [], [], []

    # warper = cv2.PyRotationWarper('plane', warped_image_scale)
    # for i, img in enumerate(conn_images):
    #     K = cameras[i].K().astype(np.float32)
    #     corner, image_wp = warper.warp(img, K, cameras[i].R,
    #                                    cv2.INTER_LINEAR, cv2.BORDER_REFLECT)

    #     corners.append(corner)
    #     sizes.append((image_wp.shape[1], image_wp.shape[0]))
    #     images_warped.append(image_wp)
    #     mask = cv2.UMat(255 * np.ones((img.shape[0], img.shape[1]), np.uint8))
    #     p, mask_wp = warper.warp(mask, K, cameras[i].R,
    #                              cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)

    #     # masks_warped.append(mask_wp.get())

    # images_warped_f = [img.astype(np.float32) for im in images_warped]

    # compensator = cv2.detail.ExposureCompensator_createDefault(
    #     cv2.detail.ExposureCompensator_NO)
    # compensator.feed(corners=corners, images=images_warped, masks=masks_warped)

    # seam_finder = cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_NO)
    # seam_finder.find(images_warped_f, corners, masks_warped)

    stitch_sizes, stitch_corners = [], []

    warper = cv2.PyRotationWarper('plane', warped_image_scale)
    for i, img in enumerate(conn_images):
        sz = img.shape[1], img.shape[0]
        K = cameras[i].K().astype(np.float32)
        roi = warper.warpRoi(sz, K, cameras[i].R)
        stitch_corners.append(roi[0:2])
        stitch_sizes.append(roi[2:4])

    canvas_size = cv2.detail.resultRoi(corners=stitch_corners,
                                       sizes=stitch_sizes)

    blend_width = np.sqrt(canvas_size[2] * canvas_size[3]) * 5 / 100
    if blend_width < 1:
        blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
    else:
        blender = cv2.detail_MultiBandBlender()
        blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int))
    blender.prepare(canvas_size)

    for i, img in enumerate(conn_images):

        K = cameras[i].K().astype(np.float32)

        corner, image_wp = warper.warp(img, K, cameras[i].R,
                                       cv2.INTER_LINEAR, cv2.BORDER_REFLECT)

        mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
        _, mask_wp = warper.warp(mask, K, cameras[i].R,
                                 cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)

        # compensator.apply(i, stitch_corners[i], image_wp, mask_wp)
        image_warped_s = image_wp.astype(np.int16)
        # image_wp = []

        # dilated_mask = cv2.dilate(masks_warped[i], None)
        # seam_mask = cv2.resize(dilated_mask,
        #                        (mask_wp.shape[1], mask_wp.shape[0]),
        #                        0,
        #                        0,
        #                        cv2.INTER_LINEAR_EXACT)
        # mask_warped = cv2.bitwise_and(seam_mask, mask_wp)
        # mask_warped = mask_wp

        blender.feed(cv2.UMat(image_warped_s), mask_wp, stitch_corners[i])

    result, result_mask = blender.blend(None, None)
    cv2.imwrite('result.jpg', result)

    zoomx = 600.0 / result.shape[1]
    dst = cv2.normalize(src=result, dst=None, alpha=255.,
                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dst = cv2.resize(dst, dsize=None, fx=zoomx, fy=zoomx)
    cv2.imwrite('dst.png', dst)
    cv2.imwrite('dst.jpeg', dst)
    cv2.imshow('panorama', dst)
    cv2.waitKey()
