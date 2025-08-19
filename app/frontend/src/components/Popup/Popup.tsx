import React from "react";
import styles from "./popup.module.css";

interface PopupProps {
  isOpen: boolean;
  onClose: () => void;
  message: string;
  type?: "success" | "error" | "info";
}

const Popup: React.FC<PopupProps> = ({ isOpen, onClose, message, type = "info" }) => {
  if (!isOpen) return null;

  return (
    <div className={styles.popupOverlay}>
      <div className={`${styles.popupBox} ${styles[type]}`}>
        <p className={styles.message}>{message}</p>
        <button className={styles.closeButton} onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  );
};

export default Popup;
