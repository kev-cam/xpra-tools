# xpra-tools install layout.
#
# Default target ('list') prints what would be installed — no changes made.
# Run 'make install' to actually copy.
# Override paths on the command line, e.g.  make install PREFIX=/opt
#
# find-xpra has a hardcoded `use lib '/usr/local/src/net-mgr/lib';` in the
# source tree; install rewrites it to NETMGR_PERL5DIR so the deployed
# binary doesn't depend on the net-mgr source tree being checked out.

PREFIX           ?= /usr/local
BINDIR           ?= $(PREFIX)/bin
NETMGR_PERL5DIR  ?= $(PREFIX)/share/perl5
DESTDIR          ?=

BINS = launch-xpra show-x11 find-xpra

INSTALL ?= install

.PHONY: list install uninstall deps help

help:
	@echo 'Targets:'
	@echo '  list       — show what would be installed (default)'
	@echo '  install    — install bin/* to $$(BINDIR) (default $(BINDIR))'
	@echo '  uninstall  — remove installed scripts'
	@echo '  deps       — check runtime dependencies'
	@echo
	@echo 'Vars (override on command line):'
	@echo '  PREFIX            ($(PREFIX))'
	@echo '  BINDIR            ($(BINDIR))'
	@echo '  NETMGR_PERL5DIR   ($(NETMGR_PERL5DIR))    — where NetMgr/*.pm lives'
	@echo '  DESTDIR           ($(DESTDIR))'

list:
	@for f in $(BINS); do echo "  bin/$$f → $(DESTDIR)$(BINDIR)/$$f"; done

install: deps
	$(INSTALL) -d $(DESTDIR)$(BINDIR)
	@for f in $(BINS); do \
	  echo "  bin/$$f → $(DESTDIR)$(BINDIR)/$$f"; \
	  sed -e "s|^use lib '/usr/local/src/net-mgr/lib';|use lib '$(NETMGR_PERL5DIR)';|" \
	      bin/$$f > $(DESTDIR)$(BINDIR)/$$f.tmp && \
	  mv $(DESTDIR)$(BINDIR)/$$f.tmp $(DESTDIR)$(BINDIR)/$$f && \
	  chmod 755 $(DESTDIR)$(BINDIR)/$$f; \
	done

uninstall:
	@for f in $(BINS); do rm -fv $(DESTDIR)$(BINDIR)/$$f; done

# --- dependency check (Debian/Ubuntu apt names) ---------------------------
# Required deps are needed at runtime; missing optionals just disable a
# sub-feature.

deps:
	@miss=""; opt_miss=""; \
	check() { \
	  if /bin/sh -c "$$1" >/dev/null 2>&1; then \
	    printf "  ok       %-20s %s\n" "$$2" "$$3"; \
	  else \
	    printf "  MISSING  %-20s %s\n" "$$2" "$$3"; \
	    miss="$$miss $$2"; \
	  fi; \
	}; \
	check_opt() { \
	  if /bin/sh -c "$$1" >/dev/null 2>&1; then \
	    printf "  ok       %-20s %s (optional)\n" "$$2" "$$3"; \
	  else \
	    printf "  optional %-20s %s\n" "$$2" "$$3"; \
	    opt_miss="$$opt_miss $$2"; \
	  fi; \
	}; \
	check 'command -v perl'  perl       'Perl interpreter'; \
	check 'command -v xpra'  xpra       'xpra (attach/start sessions)'; \
	check 'command -v ssh'   openssh-client 'ssh client (remote show-x11 probe)'; \
	check 'command -v ip'    iproute2   'ip command (subnet discovery)'; \
	check '/usr/bin/perl -MNetMgr::Client -I$(NETMGR_PERL5DIR) -e 1' \
	                         net-mgr    "NetMgr::Client at $(NETMGR_PERL5DIR) (install net-mgr)"; \
	check_opt 'command -v zenity' zenity 'zenity — needed for the find-xpra menu (text fallback: --list)'; \
	check_opt 'command -v nmap'   nmap   'nmap — fallback discovery if NET_MGR_LISTEN unset'; \
	check_opt 'command -v sudo-cat' sudo-cat 'sudo-cat — required by show-x11 to read /proc/*/cmdline'; \
	miss=$$(echo $$miss | tr ' ' '\n' | sort -u | tr '\n' ' '); \
	miss=$${miss% }; miss=$${miss# }; \
	if [ -n "$$miss" ]; then \
	  echo; echo "Missing required: $$miss"; \
	  echo "Install on Debian/Ubuntu:  sudo apt install $$miss"; \
	  exit 1; \
	fi
