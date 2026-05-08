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
APPSDIR          ?= $(PREFIX)/share/applications
DESTDIR          ?=
# Set FORCE=1 to skip the dependency prompt entirely (useful in CI or when
# you know your package manager differs from Debian/Ubuntu — e.g. Cygwin).
FORCE            ?=

BINS = launch-xpra show-x11 find-xpra find-xpra-gocryptfs xpra-helper

APPS = xpra-helper.desktop

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
	@echo '  APPSDIR           ($(APPSDIR))'
	@echo '  DESTDIR           ($(DESTDIR))'
	@echo '  FORCE             (set to 1 to skip the deps prompt)'

list:
	@for f in $(BINS); do echo "  bin/$$f → $(DESTDIR)$(BINDIR)/$$f"; done
	@for f in $(APPS); do echo "  share/applications/$$f → $(DESTDIR)$(APPSDIR)/$$f"; done

# --- shared dependency-check shell snippet --------------------------------
# Sets $$miss to the space-separated list of missing required packages and
# $$opt_miss to missing optionals.  Used by both `deps` and `install`.

define DEPS_CHECK_SH
miss=""; opt_miss=""; \
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
check 'command -v xpra || [ -x "/cygdrive/c/Program Files/Xpra/Xpra_cmd.exe" ] || [ -x "/cygdrive/c/Program Files (x86)/Xpra/Xpra_cmd.exe" ]' \
                         xpra       'xpra (attach/start sessions; on Cygwin the Windows install under Program Files is detected automatically)'; \
check 'command -v ssh'   openssh-client 'ssh client (remote show-x11 probe)'; \
check_opt 'command -v ip || command -v ifconfig || command -v ipconfig' \
                         iproute2   'ip / ifconfig / ipconfig (local-IP detection + nmap-scan source IP; any one is fine)'; \
check '/usr/bin/perl -MNetMgr::Client -I$(NETMGR_PERL5DIR) -e 1' \
                         net-mgr    "NetMgr::Client at $(NETMGR_PERL5DIR) (install net-mgr)"; \
check '/usr/bin/perl -MTk -e 1' libtk-perl 'Perl/Tk (find-xpra chooser window)'; \
check '[ -x /usr/bin/ssh-askpass ]' ssh-askpass-gnome \
                         'ssh-askpass GUI (find-xpra password prompts; any /usr/bin/ssh-askpass provider works: ssh-askpass-gnome, ssh-askpass, ksshaskpass, ...)'; \
check_opt 'command -v nmap'   nmap   'nmap — fallback discovery if NET_MGR_LISTEN unset'; \
check_opt 'command -v sudo-cat' sudo-cat 'sudo-cat — required by show-x11 to read /proc/*/cmdline'; \
check_opt 'command -v gocryptfs' gocryptfs 'gocryptfs — needed only if ~/.ssh/keys-gocryptfs is configured (find-xpra-gocryptfs mounts the cleartext at ~/.keys on demand)'; \
check_opt 'command -v fusermount' fuse 'fusermount — companion to gocryptfs; same condition as above'; \
miss=$$(echo $$miss | tr ' ' '\n' | sort -u | tr '\n' ' '); \
miss=$${miss% }; miss=$${miss# }
endef

deps:
	@$(DEPS_CHECK_SH); \
	if [ -n "$$miss" ]; then \
	  echo; echo "Missing required: $$miss"; \
	  echo "Install on Debian/Ubuntu:  sudo apt install $$miss"; \
	  exit 1; \
	fi

install:
	@$(DEPS_CHECK_SH); \
	if [ -n "$$miss" ]; then \
	  echo; echo "Missing required: $$miss"; \
	  echo "Install on Debian/Ubuntu:  sudo apt install $$miss"; \
	  if [ "$(FORCE)" = "1" ]; then \
	    echo "(FORCE=1: continuing despite missing dependencies)"; \
	  elif [ -t 0 ]; then \
	    printf "Continue install anyway? [y/N] "; \
	    read ans; \
	    case "$$ans" in [yY]*) ;; *) echo "Aborted."; exit 1 ;; esac; \
	  else \
	    echo "(non-interactive: aborting; rerun with FORCE=1 to override)" >&2; \
	    exit 1; \
	  fi; \
	fi
	@$(INSTALL) -d $(DESTDIR)$(BINDIR)
	@for f in $(BINS); do \
	  echo "  bin/$$f → $(DESTDIR)$(BINDIR)/$$f"; \
	  sed -e "s|^use lib '/usr/local/src/net-mgr/lib';|use lib '$(NETMGR_PERL5DIR)';|" \
	      bin/$$f > $(DESTDIR)$(BINDIR)/$$f.tmp && \
	  mv $(DESTDIR)$(BINDIR)/$$f.tmp $(DESTDIR)$(BINDIR)/$$f && \
	  chmod 755 $(DESTDIR)$(BINDIR)/$$f; \
	done
	@$(INSTALL) -d $(DESTDIR)$(APPSDIR)
	@for f in $(APPS); do \
	  echo "  share/applications/$$f → $(DESTDIR)$(APPSDIR)/$$f"; \
	  $(INSTALL) -m 644 share/applications/$$f $(DESTDIR)$(APPSDIR)/$$f; \
	done

uninstall:
	@for f in $(BINS); do rm -fv $(DESTDIR)$(BINDIR)/$$f; done
	@for f in $(APPS); do rm -fv $(DESTDIR)$(APPSDIR)/$$f; done
