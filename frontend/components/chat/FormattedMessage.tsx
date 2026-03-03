import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";

interface FormattedMessageProps {
  content: string;
  formatted?: string;
  type?: string;
}

export function FormattedMessage({ content, formatted }: FormattedMessageProps) {
  const displayContent = formatted || content;

  const isJson =
    !formatted &&
    (content.trim().startsWith("{") || content.trim().startsWith("["));

  if (isJson) {
    try {
      const parsed = JSON.parse(content);
      return (
        <details className="my-2 bg-secondary rounded-lg border border-border">
          <summary className="cursor-pointer px-4 py-2 font-medium text-muted-foreground hover:text-foreground text-sm">
            View Raw Data
          </summary>
          <pre className="px-4 pb-4 text-xs text-muted-foreground overflow-x-auto">
            {JSON.stringify(parsed, null, 2)}
          </pre>
        </details>
      );
    } catch { /* fall through */ }
  }

  return (
    <div className="formatted-message prose max-w-none text-foreground/80">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ node, inline, className, children, ...props }: any) {
            const match = /language-(\w+)/.exec(className || "");
            const language = match ? match[1] : "";

            return !inline ? (
              <div className="relative group">
                <SyntaxHighlighter
                  style={vscDarkPlus as any}
                  language={language || "text"}
                  PreTag="div"
                  className="rounded-lg text-sm"
                  {...props}
                >
                  {String(children).replace(/\n$/, "")}
                </SyntaxHighlighter>
                <button
                  onClick={() => navigator.clipboard.writeText(String(children))}
                  className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity px-2 py-1 bg-secondary hover:bg-muted rounded text-xs text-foreground/70"
                >
                  Copy
                </button>
              </div>
            ) : (
              <code className="bg-secondary px-1.5 py-0.5 rounded text-sm font-mono" {...props}>
                {children}
              </code>
            );
          },
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-primary/40 pl-4 py-1 my-3 text-muted-foreground italic">
              {children}
            </blockquote>
          ),
          a: ({ href, children }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline"
            >
              {children}
            </a>
          ),
        }}
      >
        {displayContent}
      </ReactMarkdown>
    </div>
  );
}
