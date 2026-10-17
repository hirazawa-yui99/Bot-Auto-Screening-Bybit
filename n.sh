sed -i '/^\s*_f = getattr(composite, "funding", None)/d' main.py

# Replace getattr(...) wrappers with direct access
sed -i \
  -e 's/getattr(_f, "rate_pct", 0.0) if _f else 0.0/composite.funding.rate_pct/g' \
  -e 's/getattr(_f, "annualized_pct", 0.0) if _f else 0.0/composite.funding.annualized_pct/g' \
  -e 's/getattr(_f, "regime", "neutral") if _f else "neutral"/composite.funding.regime/g' \
  -e 's/getattr(_f, "contrarian_bias", "neutral") if _f else "neutral"/composite.funding.contrarian_bias/g' \
  -e 's/getattr(_f, "should_block_long", False) if _f else False/composite.funding.should_block_long/g' \
  -e 's/getattr(_f, "should_block_short", False) if _f else False/composite.funding.should_block_short/g' \
  main.py

# Fix the f-string in the reason= field (most fragile replacement, do separately)
sed -i 's/getattr(_f, '"'"'rate_pct'"'"', 0\.0) if _f else 0\.0/composite.funding.rate_pct/g' main.py

# Verify no _f references remain
grep -n "_f\b" main.py || echo "No _f references remain"
