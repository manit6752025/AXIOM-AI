import type { LucideIcon } from "lucide-react";
import { cloneElement, isValidElement, useEffect, useRef, type ReactNode, type RefObject } from "react";

type FieldProps = {
  label: string;
  htmlFor: string;
  hint?: string;
  hintTone?: "default" | "success";
  error?: string;
  action?: ReactNode;
  children: ReactNode;
};

export function FormField({
  label,
  htmlFor,
  hint,
  hintTone = "default",
  error,
  action,
  children,
}: FieldProps) {
  type ControlAccessibilityProps = {
    "aria-describedby"?: string;
    "aria-invalid"?: boolean;
  };
  const control = isValidElement<ControlAccessibilityProps>(children) ? children : null;
  const describedBy = [
    control?.props["aria-describedby"],
    hint ? `${htmlFor}-hint` : undefined,
    error ? `${htmlFor}-error` : undefined,
  ].filter(Boolean).join(" ");
  const accessibleControl = control
    ? cloneElement(control, {
      "aria-describedby": describedBy || undefined,
      "aria-invalid": error ? true : control.props["aria-invalid"],
    })
    : children;

  return (
    <div className="field">
      <div className="field-label-row"><label htmlFor={htmlFor}>{label}</label>{action}</div>
      <div className="input-shell">{accessibleControl}</div>
      {hint && <small id={`${htmlFor}-hint`} className={hintTone === "success" ? "success-text" : ""}>{hint}</small>}
      {error && <small id={`${htmlFor}-error`} className="field-error" role="alert">{error}</small>}
    </div>
  );
}

type EmptyStateProps = {
  icon: LucideIcon;
  title: string;
  description: string;
  action?: ReactNode;
};

export function EmptyState({ icon: Icon, title, description, action }: EmptyStateProps) {
  return (
    <div className="empty-state" role="status" aria-live="polite">
      <div className="empty-icon" aria-hidden="true"><Icon size={20} /></div>
      <h3>{title}</h3>
      <p>{description}</p>
      {action && <div className="empty-action">{action}</div>}
    </div>
  );
}

export type DataTableColumn<Row> = {
  key: string;
  header: string;
  render: (row: Row) => ReactNode;
  align?: "left" | "center" | "right";
};

type DataTableProps<Row> = {
  caption: string;
  columns: DataTableColumn<Row>[];
  rows: Row[];
  rowKey: (row: Row) => string;
  emptyMessage?: string;
};

export function DataTable<Row>({
  caption,
  columns,
  rows,
  rowKey,
  emptyMessage = "No records to display.",
}: DataTableProps<Row>) {
  return (
    <div className="accessible-table-wrap">
      <table className="accessible-data-table">
        <caption>{caption}</caption>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key} scope="col" style={{ textAlign: column.align ?? "left" }}>
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length > 0 ? rows.map((row) => (
            <tr key={rowKey(row)}>
              {columns.map((column) => (
                <td key={column.key} style={{ textAlign: column.align ?? "left" }}>
                  {column.render(row)}
                </td>
              ))}
            </tr>
          )) : (
            <tr>
              <td className="accessible-data-table-empty" colSpan={columns.length}>{emptyMessage}</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

type AccessibleDialogProps = {
  titleId: string;
  descriptionId?: string;
  onClose: () => void;
  closeDisabled?: boolean;
  initialFocusRef?: RefObject<HTMLElement | null>;
  children: ReactNode;
};

const FOCUSABLE = [
  "a[href]",
  "button:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  "[tabindex]:not([tabindex='-1'])",
].join(",");

export function AccessibleDialog({
  titleId,
  descriptionId,
  onClose,
  closeDisabled = false,
  initialFocusRef,
  children,
}: AccessibleDialogProps) {
  const dialogRef = useRef<HTMLElement>(null);
  const closeRef = useRef(onClose);
  const disabledRef = useRef(closeDisabled);
  closeRef.current = onClose;
  disabledRef.current = closeDisabled;

  useEffect(() => {
    const dialog = dialogRef.current;
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    if (!dialog) return;

    const focusableElements = () => Array.from(dialog.querySelectorAll<HTMLElement>(FOCUSABLE))
      .filter((element) => element.getAttribute("aria-hidden") !== "true");
    const first = initialFocusRef?.current ?? focusableElements()[0] ?? dialog;
    first.focus();

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (!disabledRef.current) {
          event.preventDefault();
          closeRef.current();
        }
        return;
      }
      if (event.key !== "Tab") return;
      const elements = focusableElements();
      if (!elements.length) {
        event.preventDefault();
        dialog.focus();
        return;
      }
      const firstElement = elements[0];
      const lastElement = elements[elements.length - 1];
      if (event.shiftKey && (document.activeElement === firstElement || !dialog.contains(document.activeElement))) {
        event.preventDefault();
        lastElement.focus();
      } else if (!event.shiftKey && (document.activeElement === lastElement || !dialog.contains(document.activeElement))) {
        event.preventDefault();
        firstElement.focus();
      }
    };

    dialog.addEventListener("keydown", handleKeyDown);
    return () => {
      dialog.removeEventListener("keydown", handleKeyDown);
      if (previousFocus?.isConnected) previousFocus.focus();
    };
  }, [initialFocusRef]);

  return (
    <div className="dialog-backdrop" onMouseDown={(event) => {
      if (event.target === event.currentTarget && !disabledRef.current) closeRef.current();
    }}>
      <section
        ref={dialogRef}
        className="dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        {children}
      </section>
    </div>
  );
}
