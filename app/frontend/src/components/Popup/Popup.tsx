import React from "react";
import styles from "./Popup.module.css";

interface PopupProps {
    isOpen: boolean;
    onClose: () => void;
    children: React.ReactNode;
}

const Popup: React.FC<PopupProps> = ({ isOpen, onClose, children }) => {
    if (!isOpen) return null;

    return (
        <div className={styles.container} onClick={onClose}>
            <div
                className={styles.content}
                onClick={(e) => e.stopPropagation()} // Prevent closing when clicking inside
            >
                <button className={styles.closeButton} onClick={onClose}>
                    
                </button>
                {children}
            </div>
        </div>
    );
};

export default Popup;
