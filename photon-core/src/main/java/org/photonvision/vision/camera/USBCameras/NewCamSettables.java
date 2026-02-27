/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.vision.camera.USBCameras;

import edu.wpi.first.cscore.UsbCamera;
import org.photonvision.common.configuration.CameraConfiguration;

public class NewCamSettables extends GenericUSBCameraSettables {
    public NewCamSettables(CameraConfiguration configuration, UsbCamera camera) {
        super(configuration, camera);
    }

    // @Override
    // protected void setUpExposureProperties() {
    //     super.setUpExposureProperties();

    //     if (exposureAbsProp != null) {
    //         logger.debug("Exposure property: name=" + exposureAbsProp.getName()
    //                 + " min=" + exposureAbsProp.getMin()
    //                 + " max=" + exposureAbsProp.getMax());
    //     } else {
    //         logger.warn("No exposure property found!");
    //     }
    //     if (autoExposureProp != null) {
    //         logger.debug("Auto-exposure property: name=" + autoExposureProp.getName()
    //                 + " min=" + autoExposureProp.getMin()
    //                 + " max=" + autoExposureProp.getMax());
    //     }

    //     this.minExposure = 1;
    //     this.maxExposure = 300;
    // }

    @Override
    protected void setUpExposureProperties() {
        autoExposureProp = findProperty("exposure_auto", "auto_exposure").get();
        exposureAbsProp = findProperty("raw_exposure_time_absolute", "raw_exposure_absolute").get();

        this.minExposure = exposureAbsProp.getMin();
        this.maxExposure = exposureAbsProp.getMax();
    }

    @Override
    public void setAllCamDefaults() {
        super.setAllCamDefaults();
        logger.info("Setting All Cam Defaults :: NewCam");
        softSet("focus_automatic_continuous", 0);
        softSet("focus_absolute", 0);

        softSet("power_line_frequency", 2); // Assume 60Hz USA
        softSet("exposure_metering_mode", 0);
        softSet("exposure_dynamic_framerate", 0);
    }

    @Override
    public void setAutoExposureImpl(boolean cameraAutoExposure) {
        logger.info("Setting Auto Exposure :: NewCam");
        // softSet("focus_automatic_continuous", 0);
        super.setAutoExposureImpl(cameraAutoExposure);
        // softSet("focus_absolute", 0);
    }

    @Override
    public void setExposureRaw(double exposureRaw) {
        logger.info("NewCamSettables.setExposureRaw called with " + exposureRaw);
        // softSet("focus_automatic_continuous", 0);
        super.setExposureRaw(exposureRaw);
        // softSet("focus_automatic_continuous", 0);
        // softSet("focus_absolute", 0);
    }
}
