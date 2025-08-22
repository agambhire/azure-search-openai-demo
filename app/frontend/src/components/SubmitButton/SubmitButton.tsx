import { SendRegular } from "@fluentui/react-icons";
import { Button } from "@fluentui/react-components";
import { useTranslation } from "react-i18next";

import styles from "./SubmitButton.module.css";

interface Props {
    className?: string;
    onClick?: () => void; // made optional
    disabled?: boolean;  
}

export const SubmitButton = ({ className, onClick, disabled }: Props) => {
    const { t } = useTranslation();
    return (
        <div className={`${styles.container} ${className ?? ""}`}>
            <Button
                icon={<SendRegular />}
                onClick={onClick}
                disabled={disabled}   
            >
                {t("submit")}
            </Button>
        </div>
    );
};