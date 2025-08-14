import React from "react";
import styles from "./popup.module.css";

interface PopupProps {
    isOpen: boolean;
    onClose: () => void;
    message: string; 
}

const Popup: React.FC<PopupProps> = ({ isOpen, onClose, message }) => {
    if (!isOpen) return null;

    return (
        <div className={styles.overlay}>
            <div className={styles.popup}>
                <div className={styles.content}>
                    <p>{message}</p>
                </div>
                <button onClick={onClose} className={styles.closeBtn}>
                    Close
                </button>
            </div>
        </div>
    );
};

export default Popup;
