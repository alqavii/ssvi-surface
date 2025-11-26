import * as React from "react"
import { Check, ChevronsUpDown, Search } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"

// Mock data - In production this would come from your backend API
const tickers = [
  { value: "AAPL", label: "AAPL - Apple Inc." },
  { value: "MSFT", label: "MSFT - Microsoft Corp." },
  { value: "GOOGL", label: "GOOGL - Alphabet Inc." },
  { value: "AMZN", label: "AMZN - Amazon.com Inc." },
  { value: "TSLA", label: "TSLA - Tesla Inc." },
  { value: "NVDA", label: "NVDA - NVIDIA Corp." },
  { value: "META", label: "META - Meta Platforms Inc." },
  { value: "AMD", label: "AMD - Advanced Micro Devices" },
  { value: "SPY", label: "SPY - SPDR S&P 500 ETF Trust" },
  { value: "QQQ", label: "QQQ - Invesco QQQ Trust" },
]

interface TickerSearchProps {
  onSelect: (ticker: string) => void
  initialValue?: string
}

export function TickerSearch({ onSelect, initialValue = "" }: TickerSearchProps) {
  const [open, setOpen] = React.useState(false)
  const [value, setValue] = React.useState(initialValue)

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className="w-[250px] justify-between border-primary/20 bg-background/50 backdrop-blur-sm hover:bg-accent/50"
        >
          <div className="flex items-center gap-2 truncate">
            <Search className="h-4 w-4 text-muted-foreground" />
            {value
              ? tickers.find((t) => t.value === value)?.label
              : "Select ticker..."}
          </div>
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[250px] p-0 border-primary/20">
        <Command>
          <CommandInput placeholder="Search ticker..." />
          <CommandList>
            <CommandEmpty>No ticker found.</CommandEmpty>
            <CommandGroup>
              {tickers.map((ticker) => (
                <CommandItem
                  key={ticker.value}
                  value={ticker.value}
                  onSelect={(currentValue) => {
                    // Command component lowercases values, so we map back to our uppercase value
                    const selected = tickers.find(
                      (t) => t.value.toLowerCase() === currentValue.toLowerCase()
                    )
                    if (selected) {
                      setValue(selected.value)
                      onSelect(selected.value)
                    }
                    setOpen(false)
                  }}
                >
                  <Check
                    className={cn(
                      "mr-2 h-4 w-4",
                      value === ticker.value ? "opacity-100" : "opacity-0"
                    )}
                  />
                  {ticker.label}
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}

