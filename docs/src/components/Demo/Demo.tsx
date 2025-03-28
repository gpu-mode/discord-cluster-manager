import React from "react";
import styles from "./index.module.css";

export function Demo() {
  const videoRef = React.useRef<HTMLVideoElement>(null);

  return (
    <div
      style={{ paddingBottom: "10px", paddingTop: "10px", textAlign: "center" }}
    >
      <video
        playsInline
        autoPlay={true}
        loop
        className={styles.demo}
        muted
        onMouseOver={() => (videoRef.current.controls = true)}
        onMouseOut={() => (videoRef.current.controls = false)}
        ref={videoRef}
      >
        <source src="img/preview.mp4" type="video/mp4"></source>
      </video>
    </div>
  );
}
