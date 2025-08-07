import React, { useState, ChangeEvent } from "react";
import { Callout, Label, Text } from "@fluentui/react";
import { Button } from "@fluentui/react-components";
import { Add24Regular, Delete24Regular } from "@fluentui/react-icons";
import { useMsal } from "@azure/msal-react";
import { useTranslation } from "react-i18next";
import readNDJSONStream from "ndjson-readablestream";

import { SimpleAPIResponse, uploadFileApi, deleteUploadedFileApi, listUploadedFilesApi } from "../../api";
import { useLogin, getToken } from "../../authConfig";
import styles from "./UploadFile.module.css";

interface Props {
    className?: string;
    disabled?: boolean;
    onUploadResponse?: (response: string) => void;
    onStreamResponse?: (response: ReadableStream<any>) => void;
    shouldStream?: boolean;
}

export const UploadFile: React.FC<Props> = ({ className, disabled, onUploadResponse, onStreamResponse, shouldStream = true }: Props) => {
    // State variables to manage the component behavior
    const [isCalloutVisible, setIsCalloutVisible] = useState<boolean>(false);
    const [isUploading, setIsUploading] = useState<boolean>(false);
    const [isLoading, setIsLoading] = useState<boolean>(true);
    const [deletionStatus, setDeletionStatus] = useState<{ [filename: string]: "pending" | "error" | "success" }>({});
    const [uploadedFile, setUploadedFile] = useState<SimpleAPIResponse>();
    const [uploadedFileError, setUploadedFileError] = useState<string>();
    const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
    const { t } = useTranslation();

    if (!useLogin) {
        throw new Error("The UploadFile component requires useLogin to be true");
    }

    const client = useMsal().instance;

    // Handler for the "Manage file uploads" button
    const handleButtonClick = async () => {
        setIsCalloutVisible(!isCalloutVisible); // Toggle the Callout visibility

        // Update uploaded files by calling the API
        try {
            const idToken = await getToken(client);
            if (!idToken) {
                throw new Error("No authentication token available");
            }
            listUploadedFiles(idToken);
        } catch (error) {
            console.error(error);
            setIsLoading(false);
        }
    };

    const listUploadedFiles = async (idToken: string) => {
        listUploadedFilesApi(idToken).then(files => {
            setIsLoading(false);
            setDeletionStatus({});
            setUploadedFiles(files);
        });
    };

    const handleRemoveFile = async (filename: string) => {
        setDeletionStatus({ ...deletionStatus, [filename]: "pending" });

        try {
            const idToken = await getToken(client);
            if (!idToken) {
                throw new Error("No authentication token available");
            }

            await deleteUploadedFileApi(filename, idToken);
            setDeletionStatus({ ...deletionStatus, [filename]: "success" });
            listUploadedFiles(idToken);
        } catch (error) {
            setDeletionStatus({ ...deletionStatus, [filename]: "error" });
            console.error(error);
        }
    };

    // Function to handle streamed response data
    const handleStreamedResponse = async (response: Response) => {
        if (!response.body) {
            throw new Error("No response body available");
        }

        let accumulatedResponse = "";
        const reader = readNDJSONStream(response.body);

        try {
            for await (const data of reader) {
                if (data.error) {
                    setUploadedFileError(data.error);
                    continue;
                }

                const newContent = data.message?.content || 
                                 data.answer || 
                                 data.choices?.[0]?.message?.content ||
                                 data.message ||
                                 "";

                accumulatedResponse += newContent;
                await new Promise(resolve => setTimeout(resolve, 33)); // Throttle updates
                setUploadedFile({
                    message: accumulatedResponse
                });
                
                // Send response to chat component if callback exists
                if (onUploadResponse) {
                    onUploadResponse(accumulatedResponse);
                }
            }
        } catch (e) {
            console.warn("Failed to parse streaming response:", e);
            throw e;
        }
    };

    // Handler for the form submission (file upload)
    const handleUploadFile = async (e: ChangeEvent<HTMLInputElement>) => {
        e.preventDefault();
        if (!e.target.files || e.target.files.length === 0) {
            return;
        }
        setIsUploading(true); // Start the loading state
        const file: File = e.target.files[0];
        const formData = new FormData();
        formData.append("file", file);

        try {
            const idToken = await getToken(client);
            if (!idToken) {
                throw new Error("No authentication token available");
            }
            const response = await uploadFileApi(formData, shouldStream, idToken);
            
            if (shouldStream && response.body) {
                if (onStreamResponse) {
                    onStreamResponse(response.body);
                } else {
                    await handleStreamedResponse(response);
                }
            } else {
                const jsonResponse = await response.json();
                setUploadedFile(jsonResponse);
                if (onUploadResponse) {
                    onUploadResponse(jsonResponse.message || jsonResponse.answer || jsonResponse.choices?.[0]?.message?.content || '');
                }
            }
            
            setIsUploading(false);
            setUploadedFileError(undefined);
            listUploadedFiles(idToken);
        } catch (error) {
            console.error(error);
            setIsUploading(false);
            setUploadedFileError(t("upload.uploadedFileError"));
        }
    };

    return (
        <div className={`${styles.container} ${className ?? ""}`}>
            <div>
                <Button id="calloutButton" icon={<Add24Regular />} disabled={disabled} onClick={handleButtonClick}>
                    {t("upload.manageFileUploads")}
                </Button>

                {isCalloutVisible && (
                    <Callout
                        role="dialog"
                        gapSpace={0}
                        className={styles.callout}
                        target="#calloutButton"
                        onDismiss={() => setIsCalloutVisible(false)}
                        setInitialFocus
                    >
                        <form encType="multipart/form-data">
                            <div>
                                <Label>{t("upload.fileLabel")}</Label>
                                <input
                                    accept=".txt, .md, .json, .png, .jpg, .jpeg, .bmp, .heic, .tiff, .pdf, .docx, .xlsx, .pptx, .html"
                                    className={styles.chooseFiles}
                                    type="file"
                                    onChange={handleUploadFile}
                                />
                            </div>
                        </form>

                        {/* Show a loading message while files are being uploaded */}
                        {isUploading && <Text>{t("upload.uploadingFiles")}</Text>}
                        {!isUploading && uploadedFileError && <Text>{uploadedFileError}</Text>}
                        {!isUploading && uploadedFile && (
                            <Text>{uploadedFile.message || uploadedFile.answer || (uploadedFile.choices?.[0]?.message?.content)}</Text>
                        )}

                        {/* Display the list of already uploaded */}
                        <h3>{t("upload.uploadedFilesLabel")}</h3>

                        {isLoading && <Text>{t("upload.loading")}</Text>}
                        {!isLoading && uploadedFiles.length === 0 && <Text>{t("upload.noFilesUploaded")}</Text>}
                        {uploadedFiles.length >= 5 ? (
                            <div className={styles.uploadedFilesScroll}>
                                {uploadedFiles.map((filename, index) => (
                                    <div key={index} className={styles.list}>
                                        <div className={styles.item}>{filename}</div>
                                        <Button
                                            icon={<Delete24Regular />}
                                            onClick={() => handleRemoveFile(filename)}
                                            disabled={deletionStatus[filename] === "pending" || deletionStatus[filename] === "success"}
                                        >
                                            {!deletionStatus[filename] && t("upload.deleteFile")}
                                            {deletionStatus[filename] == "pending" && t("upload.deletingFile")}
                                            {deletionStatus[filename] == "error" && t("upload.errorDeleting")}
                                            {deletionStatus[filename] == "success" && t("upload.fileDeleted")}
                                        </Button>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <>
                                {uploadedFiles.map((filename, index) => (
                                    <div key={index} className={styles.list}>
                                        <div className={styles.item}>{filename}</div>
                                        <Button
                                            icon={<Delete24Regular />}
                                            onClick={() => handleRemoveFile(filename)}
                                            disabled={deletionStatus[filename] === "pending" || deletionStatus[filename] === "success"}
                                        >
                                            {!deletionStatus[filename] && t("upload.deleteFile")}
                                            {deletionStatus[filename] == "pending" && t("upload.deletingFile")}
                                            {deletionStatus[filename] == "error" && t("upload.errorDeleting")}
                                            {deletionStatus[filename] == "success" && t("upload.fileDeleted")}
                                        </Button>
                                    </div>
                                ))}
                            </>
                        )}
                    </Callout>
                )}
            </div>
        </div>
    );
};